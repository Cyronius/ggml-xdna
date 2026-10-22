const button = document.getElementById('copy-url');
const status = document.getElementById('copy-status');
let resetCopy;
button?.addEventListener('click', async () => {
  const value = document.getElementById('base-url').textContent;
  try {
    await navigator.clipboard.writeText(value);
    clearTimeout(resetCopy);
    button.textContent = 'Copied';
    status.textContent = 'Lemonade Base URL copied.';
    resetCopy = setTimeout(() => { button.textContent = 'Copy'; }, 2400);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('base-url'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Copy was blocked. The address is selected; press Ctrl+C or use your browser’s copy command.';
    button.textContent = 'Select URL';
  }
});
