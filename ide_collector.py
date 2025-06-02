"""
IDE activity collector for tracking coding behavior and patterns.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class IDECollector:
    def __init__(self, workspace_path: str):
        """Initialize the IDE activity collector.
        
        Args:
            workspace_path: Path to the workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self.activity_log = []
        self.current_file = None
        self.session_start = datetime.now()
        
    def track_file_open(self, file_path: str) -> None:
        """Track when a file is opened.
        
        Args:
            file_path: Path to the opened file
        """
        self.current_file = file_path
        self._log_activity('file_open', {
            'file_path': file_path,
            'timestamp': datetime.now().isoformat()
        })
        
    def track_edits(self, file_path: str, changes: List[Dict]) -> None:
        """Track code edits in a file.
        
        Args:
            file_path: Path to the edited file
            changes: List of changes made to the file
        """
        self._log_activity('code_edit', {
            'file_path': file_path,
            'changes': changes,
            'timestamp': datetime.now().isoformat()
        })
        
    def track_debug_session(self, file_path: str, debug_data: Dict) -> None:
        """Track debugging sessions.
        
        Args:
            file_path: Path to the file being debugged
            debug_data: Debug session information
        """
        self._log_activity('debug_session', {
            'file_path': file_path,
            'debug_data': debug_data,
            'timestamp': datetime.now().isoformat()
        })
        
    def track_refactoring(self, file_path: str, refactoring_data: Dict) -> None:
        """Track code refactoring operations.
        
        Args:
            file_path: Path to the refactored file
            refactoring_data: Refactoring operation details
        """
        self._log_activity('refactoring', {
            'file_path': file_path,
            'refactoring_data': refactoring_data,
            'timestamp': datetime.now().isoformat()
        })
        
    def track_shortcuts(self, shortcut: str, context: Dict) -> None:
        """Track IDE shortcut usage.
        
        Args:
            shortcut: The shortcut used
            context: Context information about the shortcut usage
        """
        self._log_activity('shortcut', {
            'shortcut': shortcut,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        
    def get_coding_patterns(self) -> Dict:
        """Analyze and return coding patterns from collected data.
        
        Returns:
            Dictionary containing coding pattern statistics
        """
        return {
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'files_worked_on': self._get_files_worked_on(),
            'edit_patterns': self._analyze_edit_patterns(),
            'debug_patterns': self._analyze_debug_patterns(),
            'refactoring_patterns': self._analyze_refactoring_patterns(),
            'shortcut_usage': self._analyze_shortcut_usage()
        }
        
    def save_activity_log(self, output_path: str) -> None:
        """Save the activity log to a file.
        
        Args:
            output_path: Path to save the activity log
        """
        with open(output_path, 'w') as f:
            json.dump(self.activity_log, f, indent=2)
            
    def _log_activity(self, activity_type: str, data: Dict) -> None:
        """Log an activity event.
        
        Args:
            activity_type: Type of activity
            data: Activity data
        """
        self.activity_log.append({
            'type': activity_type,
            'data': data
        })
        
    def _get_files_worked_on(self) -> List[str]:
        """Get list of files worked on during the session."""
        return list(set(
            activity['data']['file_path']
            for activity in self.activity_log
            if 'file_path' in activity['data']
        ))
        
    def _analyze_edit_patterns(self) -> Dict:
        """Analyze code editing patterns."""
        edit_activities = [
            activity for activity in self.activity_log
            if activity['type'] == 'code_edit'
        ]
        return {
            'total_edits': len(edit_activities),
            'files_edited': len(set(
                activity['data']['file_path']
                for activity in edit_activities
            ))
        }
        
    def _analyze_debug_patterns(self) -> Dict:
        """Analyze debugging patterns."""
        debug_activities = [
            activity for activity in self.activity_log
            if activity['type'] == 'debug_session'
        ]
        return {
            'total_debug_sessions': len(debug_activities),
            'files_debugged': len(set(
                activity['data']['file_path']
                for activity in debug_activities
            ))
        }
        
    def _analyze_refactoring_patterns(self) -> Dict:
        """Analyze refactoring patterns."""
        refactoring_activities = [
            activity for activity in self.activity_log
            if activity['type'] == 'refactoring'
        ]
        return {
            'total_refactorings': len(refactoring_activities),
            'files_refactored': len(set(
                activity['data']['file_path']
                for activity in refactoring_activities
            ))
        }
        
    def _analyze_shortcut_usage(self) -> Dict:
        """Analyze shortcut usage patterns."""
        shortcut_activities = [
            activity for activity in self.activity_log
            if activity['type'] == 'shortcut'
        ]
        shortcut_counts = {}
        for activity in shortcut_activities:
            shortcut = activity['data']['shortcut']
            shortcut_counts[shortcut] = shortcut_counts.get(shortcut, 0) + 1
        return shortcut_counts 