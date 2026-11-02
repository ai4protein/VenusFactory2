import { useEffect, useRef } from 'react';
import { bindFileTransfer, type FileTransferOptions } from './fileTransfer';

export function useFileTransfer(target: HTMLElement | null | undefined, options: FileTransferOptions, active = true) {
  const latest = useRef(options); latest.current = options;
  useEffect(() => {
    if (!target || !active) return;
    return bindFileTransfer(target, () => latest.current);
  }, [target, active]);
}
