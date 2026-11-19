interface PanelContext {
  panelElement: HTMLElement;
}

interface ExtensionContext {
  registerPanel(panel: { name: string; initPanel: (context: PanelContext) => () => void }): void;
}

export function activate(extensionContext: ExtensionContext): void {
  extensionContext.registerPanel({
    name: "SO101 operator",
    initPanel(context) {
      const root = document.createElement("div");
      root.style.cssText = "height:100%;display:flex;flex-direction:column;background:#070b14;color:#e2e8f0;font:12px sans-serif";
      const frame = document.createElement("iframe");
      frame.title = "SO101 Suite operator controls";
      frame.src = "http://127.0.0.1:8000/";
      frame.style.cssText = "flex:1;min-height:0;width:100%;border:0";
      const keys = document.createElement("p");
      keys.textContent = "H / Escape work when the Suite panel has focus. Use the visible Suite stop buttons or physical cutoff otherwise.";
      keys.style.cssText = "margin:0;padding:6px 8px;background:#422006;color:#fde68a";
      root.append(keys, frame);
      context.panelElement.append(root);
      return () => { frame.src = "about:blank"; root.remove(); };
    },
  });
}
