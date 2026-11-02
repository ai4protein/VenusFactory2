import { useEffect, type RefObject } from "react";

/** Dismiss a nonmodal disclosure without stealing focus from an outside click. */
export function useDismissibleDetails(ref: RefObject<HTMLDetailsElement>, visible = true) {
  useEffect(() => {
    if (!visible && ref.current) ref.current.open = false;
    function pointer(event: PointerEvent) {
      const details = ref.current;
      if (details?.open && event.target instanceof Node && !details.contains(event.target)) details.open = false;
    }
    function keyboard(event: KeyboardEvent) {
      const details = ref.current;
      if (event.key === "Escape" && details?.open) {
        details.open = false;
        details.querySelector("summary")?.focus();
      }
    }
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("pointerdown", pointer);
      document.removeEventListener("keydown", keyboard);
    };
  }, [ref, visible]);
}
