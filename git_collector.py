"""
Git data collector for gathering coding history and patterns.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

from git import Repo, Commit
import pandas as pd

class GitDataCollector:
    def __init__(self, repo_path: str):
        """Initialize the Git data collector.
        
        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        
    def collect_commit_history(self, since_date: Optional[datetime] = None) -> pd.DataFrame:
        """Collect commit history and metadata.
        
        Args:
            since_date: Optional date to filter commits from
            
        Returns:
            DataFrame containing commit history
        """
        commits_data = []
        
        for commit in self.repo.iter_commits():
            if since_date and commit.committed_datetime < since_date:
                continue
                
            commit_data = {
                'hash': commit.hexsha,
                'author': commit.author.name,
                'email': commit.author.email,
                'date': commit.committed_datetime,
                'message': commit.message,
                'files_changed': len(commit.stats.files),
                'insertions': commit.stats.total['insertions'],
                'deletions': commit.stats.total['deletions']
            }
            commits_data.append(commit_data)
            
        return pd.DataFrame(commits_data)
    
    def collect_file_changes(self, commit: Commit) -> List[Dict]:
        """Collect detailed file changes for a specific commit.
        
        Args:
            commit: Git commit object
            
        Returns:
            List of dictionaries containing file change details
        """
        changes = []
        
        for diff in commit.diff(commit.parents[0] if commit.parents else None):
            change = {
                'file_path': diff.a_path,
                'change_type': diff.change_type,
                'lines_added': diff.stats.get('insertions', 0),
                'lines_deleted': diff.stats.get('deletions', 0),
                'diff': diff.diff.decode('utf-8') if diff.diff else None
            }
            changes.append(change)
            
        return changes
    
    def collect_coding_patterns(self) -> Dict:
        """Collect coding patterns and statistics.
        
        Returns:
            Dictionary containing coding pattern statistics
        """
        patterns = {
            'commit_frequency': self._analyze_commit_frequency(),
            'file_changes': self._analyze_file_changes(),
            'time_patterns': self._analyze_time_patterns()
        }
        return patterns
    
    def _analyze_commit_frequency(self) -> Dict:
        """Analyze commit frequency patterns."""
        commits = list(self.repo.iter_commits())
        return {
            'total_commits': len(commits),
            'avg_commits_per_day': len(commits) / max(1, (commits[0].committed_datetime - commits[-1].committed_datetime).days)
        }
    
    def _analyze_file_changes(self) -> Dict:
        """Analyze file change patterns."""
        file_changes = {}
        for commit in self.repo.iter_commits():
            for diff in commit.diff(commit.parents[0] if commit.parents else None):
                file_changes[diff.a_path] = file_changes.get(diff.a_path, 0) + 1
        return file_changes
    
    def _analyze_time_patterns(self) -> Dict:
        """Analyze coding time patterns."""
        commits = list(self.repo.iter_commits())
        hours = [commit.committed_datetime.hour for commit in commits]
        return {
            'most_active_hour': max(set(hours), key=hours.count),
            'avg_commits_per_hour': len(commits) / 24
        } 