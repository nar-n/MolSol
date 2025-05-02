import * as vscode from 'vscode';
import { MarkdownRenderer } from './markdownRenderer';

export function activate(context: vscode.ExtensionContext) {
    const markdownRenderer = new MarkdownRenderer();

    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(event => {
            if (event.document.languageId === 'markdown') {
                const markdownContent = event.document.getText();
                const renderedHtml = markdownRenderer.renderMarkdown(markdownContent);
                // Logic to display the rendered HTML in the notebook or editor
            }
        })
    );
}

export function deactivate() {}