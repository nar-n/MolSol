export function sanitizeMarkdown(input: string): string {
    const element = document.createElement('div');
    element.innerText = input;
    return element.innerHTML;
}

export function isValidMarkdown(input: string): boolean {
    return typeof input === 'string' && input.trim().length > 0;
}