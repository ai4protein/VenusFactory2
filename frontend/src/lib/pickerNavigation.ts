export type PickerItem = { value: string; label: string; disabled?: boolean; hint?: string };

export function nextPickerItem(items: PickerItem[], current: number, direction: 1 | -1): number {
  for (let offset = 1; offset <= items.length; offset++) {
    const index = (current + direction * offset + items.length) % items.length;
    if (!items[index].disabled) return index;
  }
  return -1;
}

export function matchPickerItem(items: PickerItem[], query: string, current: number): number {
  for (let offset = 1; offset <= items.length; offset++) {
    const index = (current + offset + items.length) % items.length;
    if (!items[index].disabled && items[index].label.toLocaleLowerCase().startsWith(query.toLocaleLowerCase())) return index;
  }
  return -1;
}

export function pickerPlacement(rect: { left: number; top: number; bottom: number; width: number }, viewport: { width: number; height: number }) {
  const width = Math.min(296, Math.max(0, viewport.width - 24));
  const above = rect.top - 20;
  const below = viewport.height - rect.bottom - 20;
  const upward = above >= Math.min(320, below);
  return {
    width,
    left: Math.max(12, Math.min(rect.left, viewport.width - width - 12)),
    maxHeight: Math.max(0, Math.min(360, upward ? above : below)),
    ...(upward ? { bottom: viewport.height - rect.top + 6 } : { top: rect.bottom + 6 }),
  };
}
