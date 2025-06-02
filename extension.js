"""
VS Code extension for the digital twin.
"""
const vscode = require('vscode');
const axios = require('axios');

// Configuration
const API_URL = 'http://localhost:8000';
const CONFIG_KEY = 'codebrainDigitalTwin';

// Extension activation
function activate(context) {
    console.log('CodeBrain Digital Twin extension is now active!');

    // Register commands
    let generateCode = vscode.commands.registerCommand('codebrain.generateCode', async () => {
        try {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            // Get selected text or current line
            const selection = editor.selection;
            const text = selection.isEmpty
                ? editor.document.lineAt(selection.active.line).text
                : editor.document.getText(selection);

            // Show input box for style constraints
            const styleConstraints = await vscode.window.showInputBox({
                prompt: 'Enter style constraints (JSON)',
                placeHolder: '{"indentation": 4, "naming": "snake_case"}'
            });

            // Show input box for pattern constraints
            const patternConstraints = await vscode.window.showInputBox({
                prompt: 'Enter pattern constraints (JSON)',
                placeHolder: '{"error_handling": true, "optimization": "high"}'
            });

            // Prepare request
            const request = {
                prompt: text,
                style_constraints: styleConstraints ? JSON.parse(styleConstraints) : null,
                pattern_constraints: patternConstraints ? JSON.parse(patternConstraints) : null
            };

            // Call API
            const response = await axios.post(`${API_URL}/generate`, request);

            // Insert generated code
            editor.edit(editBuilder => {
                if (selection.isEmpty) {
                    // Replace current line
                    const line = editor.document.lineAt(selection.active.line);
                    editBuilder.replace(line.range, response.data.code);
                } else {
                    // Replace selection
                    editBuilder.replace(selection, response.data.code);
                }
            });

            // Show metrics
            vscode.window.showInformationMessage(
                `Generated code with confidence: ${response.data.confidence}`
            );

        } catch (error) {
            vscode.window.showErrorMessage(`Error: ${error.message}`);
        }
    });

    // Register code completion provider
    let completionProvider = vscode.languages.registerCompletionItemProvider(
        { scheme: 'file' },
        {
            async provideCompletionItems(document, position, token, context) {
                try {
                    // Get current line
                    const line = document.lineAt(position.line).text;
                    
                    // Call API for suggestions
                    const response = await axios.post(`${API_URL}/generate`, {
                        prompt: line,
                        max_length: 50
                    });

                    // Create completion items
                    const completionItems = response.data.code
                        .split('\n')
                        .map(line => {
                            const item = new vscode.CompletionItem(line);
                            item.kind = vscode.CompletionItemKind.Snippet;
                            return item;
                        });

                    return completionItems;

                } catch (error) {
                    console.error('Completion error:', error);
                    return [];
                }
            }
        }
    );

    // Register code action provider
    let codeActionProvider = vscode.languages.registerCodeActionProvider(
        { scheme: 'file' },
        {
            async provideCodeActions(document, range, context, token) {
                try {
                    const actions = [];

                    // Add "Generate Implementation" action
                    const generateAction = new vscode.CodeAction(
                        'Generate Implementation',
                        vscode.CodeActionKind.QuickFix
                    );
                    generateAction.command = {
                        command: 'codebrain.generateCode',
                        title: 'Generate Implementation'
                    };
                    actions.push(generateAction);

                    // Add "Optimize Code" action
                    const optimizeAction = new vscode.CodeAction(
                        'Optimize Code',
                        vscode.CodeActionKind.Refactor
                    );
                    optimizeAction.command = {
                        command: 'codebrain.optimizeCode',
                        title: 'Optimize Code'
                    };
                    actions.push(optimizeAction);

                    return actions;

                } catch (error) {
                    console.error('Code action error:', error);
                    return [];
                }
            }
        }
    );

    // Register hover provider
    let hoverProvider = vscode.languages.registerHoverProvider(
        { scheme: 'file' },
        {
            async provideHover(document, position, token) {
                try {
                    // Get word at position
                    const wordRange = document.getWordRangeAtPosition(position);
                    if (!wordRange) return null;

                    const word = document.getText(wordRange);

                    // Call API for explanation
                    const response = await axios.post(`${API_URL}/generate`, {
                        prompt: `Explain this code: ${word}`,
                        max_length: 100
                    });

                    return new vscode.Hover(response.data.code);

                } catch (error) {
                    console.error('Hover error:', error);
                    return null;
                }
            }
        }
    );

    // Register status bar item
    let statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = "$(code) Digital Twin";
    statusBarItem.tooltip = "CodeBrain Digital Twin";
    statusBarItem.command = 'codebrain.showStatus';
    statusBarItem.show();

    // Add to subscriptions
    context.subscriptions.push(
        generateCode,
        completionProvider,
        codeActionProvider,
        hoverProvider,
        statusBarItem
    );
}

// Extension deactivation
function deactivate() {
    console.log('CodeBrain Digital Twin extension is now deactivated');
}

module.exports = {
    activate,
    deactivate
}; 