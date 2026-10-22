const button = document.getElementById("copy-url");
const status = document.getElementById("copy-status");
let resetCopy;
button?.addEventListener("click", async () => {
  const value = document.getElementById("base-url").textContent;
  try {
    await navigator.clipboard.writeText(value);
    clearTimeout(resetCopy);
    button.textContent = "Copied";
    status.textContent = "Lemonade Base URL copied.";
    resetCopy = setTimeout(() => {
      button.textContent = "Copy";
    }, 2400);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.getElementById("base-url"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent =
      "Copy was blocked. The address is selected; press Ctrl+C or use your browser’s copy command.";
    button.textContent = "Select URL";
  }
});

const heroComparison = document.getElementById("hero-comparison");
const toggleComparison = document.getElementById("toggle-comparison");
if (heroComparison && toggleComparison) {
  const still = heroComparison.getAttribute("src");
  const animation = heroComparison.dataset.animation;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let paused = reducedMotion.matches;
  let visible = false;

  const updateComparison = () => {
    const source = !paused && visible && !document.hidden ? animation : still;
    if (heroComparison.getAttribute("src") !== source) {
      heroComparison.src = source;
    }
    toggleComparison.textContent = paused ? "Play animation" : "Pause animation";
  };

  toggleComparison.hidden = false;
  toggleComparison.addEventListener("click", () => {
    paused = !paused;
    updateComparison();
  });
  reducedMotion.addEventListener("change", (event) => {
    paused = event.matches;
    updateComparison();
  });
  document.addEventListener("visibilitychange", updateComparison);
  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    updateComparison();
  });
  observer.observe(heroComparison);
  updateComparison();
}
