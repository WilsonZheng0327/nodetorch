// Tutorial auto-detect event bus. Lives in its own module (not TutorialPanel.tsx)
// so the panel file exports only its component and stays Fast-Refresh-friendly —
// see react-refresh/only-export-components.

/** Dispatch a tutorial auto-detect event. Call this from other components. */
export function tutorialEvent(key: string) {
  window.dispatchEvent(new CustomEvent('nodetorch-tutorial', { detail: key }));
}
