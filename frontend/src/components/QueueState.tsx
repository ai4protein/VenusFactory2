import { useLang } from '../lib/i18n';
import './JobHistory.css';
export type QueueStatus = { id: string; kind?: string; resource: string; status: string; position: number; cancel_requested?: boolean; message?: string; created?: number };
export function QueueState({ queue }: { queue?: QueueStatus | null }) {
  const { lang } = useLang(); const zh=lang==='zh';
  if (!queue || !['queued','running'].includes(queue.status) || (queue.status!=='queued' && !queue.cancel_requested)) return null;
  return <p className="job-queue-state" role="status">{queue.cancel_requested ? (zh?'正在停止…':'Stopping…') : <>{zh?'等待计算资源':'Waiting for resources'}{queue.position>0 && <> · {zh?'队列位置':'Queue position'} {queue.position}</>}</>}</p>;
}
