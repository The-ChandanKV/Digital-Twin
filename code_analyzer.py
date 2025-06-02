"""
Code analyzer for extracting features and patterns from collected data.
"""
import ast
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

class CodeAnalyzer:
    def __init__(self):
        """Initialize the code analyzer."""
        self.style_metrics = {}
        self.problem_solving_patterns = {}
        self.decision_patterns = {}
        
    def analyze_code_style(self, code: str) -> Dict:
        """Analyze code style and formatting patterns.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing style metrics
        """
        try:
            tree = ast.parse(code)
            metrics = {
                'indentation': self._analyze_indentation(code),
                'naming_conventions': self._analyze_naming_conventions(tree),
                'comment_style': self._analyze_comments(code),
                'line_length': self._analyze_line_length(code),
                'import_style': self._analyze_imports(tree)
            }
            self.style_metrics.update(metrics)
            return metrics
        except SyntaxError:
            return {}
            
    def analyze_problem_solving(self, code: str) -> Dict:
        """Analyze problem-solving patterns in the code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing problem-solving patterns
        """
        try:
            tree = ast.parse(code)
            patterns = {
                'algorithm_choices': self._analyze_algorithm_choices(tree),
                'error_handling': self._analyze_error_handling(tree),
                'code_organization': self._analyze_code_organization(tree),
                'optimization_patterns': self._analyze_optimizations(tree)
            }
            self.problem_solving_patterns.update(patterns)
            return patterns
        except SyntaxError:
            return {}
            
    def analyze_decision_making(self, code: str) -> Dict:
        """Analyze decision-making patterns in the code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing decision-making patterns
        """
        try:
            tree = ast.parse(code)
            patterns = {
                'control_flow': self._analyze_control_flow(tree),
                'design_patterns': self._analyze_design_patterns(tree),
                'abstraction_levels': self._analyze_abstraction_levels(tree),
                'code_reuse': self._analyze_code_reuse(tree)
            }
            self.decision_patterns.update(patterns)
            return patterns
        except SyntaxError:
            return {}
            
    def _analyze_indentation(self, code: str) -> Dict:
        """Analyze indentation patterns."""
        lines = code.split('\n')
        indent_sizes = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_sizes.append(indent)
                
        return {
            'avg_indent_size': sum(indent_sizes) / len(indent_sizes) if indent_sizes else 0,
            'indent_consistency': len(set(indent_sizes)) == 1 if indent_sizes else True
        }
        
    def _analyze_naming_conventions(self, tree: ast.AST) -> Dict:
        """Analyze naming convention patterns."""
        conventions = {
            'class_names': [],
            'function_names': [],
            'variable_names': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                conventions['class_names'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                conventions['function_names'].append(node.name)
            elif isinstance(node, ast.Name):
                conventions['variable_names'].append(node.id)
                
        return {
            'class_naming': self._check_naming_style(conventions['class_names'], 'class'),
            'function_naming': self._check_naming_style(conventions['function_names'], 'function'),
            'variable_naming': self._check_naming_style(conventions['variable_names'], 'variable')
        }
        
    def _analyze_comments(self, code: str) -> Dict:
        """Analyze comment style and patterns."""
        lines = code.split('\n')
        comments = [line for line in lines if line.strip().startswith('#')]
        
        return {
            'comment_ratio': len(comments) / len(lines) if lines else 0,
            'docstring_usage': bool(ast.get_docstring(ast.parse(code))),
            'inline_comments': len([c for c in comments if c.strip().startswith('# ')])
        }
        
    def _analyze_line_length(self, code: str) -> Dict:
        """Analyze line length patterns."""
        lines = code.split('\n')
        line_lengths = [len(line) for line in lines]
        
        return {
            'avg_line_length': sum(line_lengths) / len(line_lengths) if line_lengths else 0,
            'max_line_length': max(line_lengths) if line_lengths else 0,
            'long_lines_ratio': len([l for l in line_lengths if l > 80]) / len(line_lengths) if line_lengths else 0
        }
        
    def _analyze_imports(self, tree: ast.AST) -> Dict:
        """Analyze import style and patterns."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.extend(f"{node.module}.{n.name}" for n in node.names)
                
        return {
            'import_count': len(imports),
            'import_style': 'absolute' if all('.' in imp for imp in imports) else 'relative',
            'standard_library_ratio': len([imp for imp in imports if imp.split('.')[0] in ['os', 'sys', 'math']]) / len(imports) if imports else 0
        }
        
    def _analyze_algorithm_choices(self, tree: ast.AST) -> Dict:
        """Analyze algorithm and data structure choices."""
        patterns = {
            'recursion_usage': False,
            'loop_patterns': [],
            'data_structures': set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                patterns['recursion_usage'] = any(
                    isinstance(n, ast.Call) and n.func.id == node.name
                    for n in ast.walk(node)
                )
            elif isinstance(node, ast.For):
                patterns['loop_patterns'].append('for')
            elif isinstance(node, ast.While):
                patterns['loop_patterns'].append('while')
            elif isinstance(node, ast.ListComp):
                patterns['data_structures'].add('list')
            elif isinstance(node, ast.DictComp):
                patterns['data_structures'].add('dict')
                
        return patterns
        
    def _analyze_error_handling(self, tree: ast.AST) -> Dict:
        """Analyze error handling patterns."""
        try_blocks = 0
        except_blocks = 0
        finally_blocks = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_blocks += 1
                except_blocks += len(node.handlers)
                if node.finalbody:
                    finally_blocks += 1
                    
        return {
            'try_except_ratio': except_blocks / try_blocks if try_blocks else 0,
            'finally_usage': finally_blocks / try_blocks if try_blocks else 0,
            'error_handling_density': try_blocks / len(list(ast.walk(tree))) if tree else 0
        }
        
    def _analyze_code_organization(self, tree: ast.AST) -> Dict:
        """Analyze code organization patterns."""
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)
                
        return {
            'function_count': len(functions),
            'class_count': len(classes),
            'avg_function_length': sum(len(list(ast.walk(f))) for f in functions) / len(functions) if functions else 0,
            'avg_class_length': sum(len(list(ast.walk(c))) for c in classes) / len(classes) if classes else 0
        }
        
    def _analyze_optimizations(self, tree: ast.AST) -> Dict:
        """Analyze code optimization patterns."""
        optimizations = {
            'list_comprehensions': 0,
            'generator_expressions': 0,
            'set_operations': 0,
            'builtin_functions': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                optimizations['list_comprehensions'] += 1
            elif isinstance(node, ast.GeneratorExp):
                optimizations['generator_expressions'] += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['set', 'frozenset']:
                    optimizations['set_operations'] += 1
                elif node.func.id in ['map', 'filter', 'zip']:
                    optimizations['builtin_functions'] += 1
                    
        return optimizations
        
    def _analyze_control_flow(self, tree: ast.AST) -> Dict:
        """Analyze control flow patterns."""
        patterns = {
            'if_else_ratio': 0,
            'switch_case_usage': 0,
            'early_returns': 0,
            'nested_conditions': 0
        }
        
        if_count = 0
        else_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if_count += 1
                if node.orelse:
                    else_count += 1
                if isinstance(node.test, ast.BoolOp):
                    patterns['nested_conditions'] += 1
            elif isinstance(node, ast.Return) and node.value:
                patterns['early_returns'] += 1
                
        patterns['if_else_ratio'] = else_count / if_count if if_count else 0
        return patterns
        
    def _analyze_design_patterns(self, tree: ast.AST) -> Dict:
        """Analyze design pattern usage."""
        patterns = {
            'singleton': 0,
            'factory': 0,
            'observer': 0,
            'decorator': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for singleton pattern
                if any(isinstance(n, ast.ClassDef) and n.name == '__new__' for n in ast.walk(node)):
                    patterns['singleton'] += 1
                # Check for factory pattern
                if any(isinstance(n, ast.FunctionDef) and 'create' in n.name.lower() for n in ast.walk(node)):
                    patterns['factory'] += 1
                    
        return patterns
        
    def _analyze_abstraction_levels(self, tree: ast.AST) -> Dict:
        """Analyze code abstraction levels."""
        return {
            'inheritance_depth': self._calculate_inheritance_depth(tree),
            'interface_usage': self._count_interfaces(tree),
            'abstraction_ratio': self._calculate_abstraction_ratio(tree)
        }
        
    def _analyze_code_reuse(self, tree: ast.AST) -> Dict:
        """Analyze code reuse patterns."""
        return {
            'function_reuse': self._analyze_function_reuse(tree),
            'class_reuse': self._analyze_class_reuse(tree),
            'module_imports': self._analyze_module_imports(tree)
        }
        
    def _check_naming_style(self, names: List[str], style: str) -> Dict:
        """Check naming style consistency."""
        if not names:
            return {'consistent': True, 'style': 'unknown'}
            
        if style == 'class':
            is_consistent = all(name[0].isupper() for name in names)
            detected_style = 'PascalCase' if is_consistent else 'mixed'
        elif style == 'function':
            is_consistent = all(name[0].islower() for name in names)
            detected_style = 'snake_case' if is_consistent else 'mixed'
        else:  # variable
            is_consistent = all(name[0].islower() for name in names)
            detected_style = 'snake_case' if is_consistent else 'mixed'
            
        return {
            'consistent': is_consistent,
            'style': detected_style
        }
        
    def _calculate_inheritance_depth(self, tree: ast.AST) -> int:
        """Calculate maximum inheritance depth."""
        max_depth = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                depth = 0
                current = node
                while current.bases:
                    depth += 1
                    current = current.bases[0]
                max_depth = max(max_depth, depth)
        return max_depth
        
    def _count_interfaces(self, tree: ast.AST) -> int:
        """Count interface-like classes."""
        interface_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if all(isinstance(n, ast.FunctionDef) and not n.body for n in ast.walk(node)):
                    interface_count += 1
        return interface_count
        
    def _calculate_abstraction_ratio(self, tree: ast.AST) -> float:
        """Calculate ratio of abstract to concrete implementations."""
        abstract_count = 0
        concrete_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(isinstance(n, ast.FunctionDef) and not n.body for n in ast.walk(node)):
                    abstract_count += 1
                else:
                    concrete_count += 1
        return abstract_count / (abstract_count + concrete_count) if (abstract_count + concrete_count) > 0 else 0
        
    def _analyze_function_reuse(self, tree: ast.AST) -> Dict:
        """Analyze function reuse patterns."""
        function_calls = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                function_calls[node.func.id] = function_calls.get(node.func.id, 0) + 1
                
        return {
            'most_called': max(function_calls.items(), key=lambda x: x[1])[0] if function_calls else None,
            'reuse_ratio': len([c for c in function_calls.values() if c > 1]) / len(function_calls) if function_calls else 0
        }
        
    def _analyze_class_reuse(self, tree: ast.AST) -> Dict:
        """Analyze class reuse patterns."""
        class_instances = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                class_instances[node.func.id] = class_instances.get(node.func.id, 0) + 1
                
        return {
            'most_instantiated': max(class_instances.items(), key=lambda x: x[1])[0] if class_instances else None,
            'reuse_ratio': len([c for c in class_instances.values() if c > 1]) / len(class_instances) if class_instances else 0
        }
        
    def _analyze_module_imports(self, tree: ast.AST) -> Dict:
        """Analyze module import patterns."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.extend(f"{node.module}.{n.name}" for n in node.names)
                
        return {
            'import_count': len(imports),
            'unique_modules': len(set(imp.split('.')[0] for imp in imports)),
            'third_party_ratio': len([imp for imp in imports if not imp.startswith(('os', 'sys', 'math'))]) / len(imports) if imports else 0
        } 