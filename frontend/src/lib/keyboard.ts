/** True when a key press is aimed at a text field, so letter shortcuts must not fire. */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  // jsdom does not implement isContentEditable; honour the attribute directly.
  const editable = target.closest("[contenteditable]");
  if (editable && editable.getAttribute("contenteditable") !== "false") return true;
  if (target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true;
  if (target instanceof HTMLInputElement) {
    const nonText = ["button", "checkbox", "radio", "range", "reset", "submit", "color", "file", "image"];
    return !nonText.includes(target.type);
  }
  return false;
}
