import DOMPurify from "dompurify";
import { marked } from "marked";
import type { ChatHistoryItem } from "../lib/api";

const DEFAULT_AVATAR = "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
const USER_AVATAR =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#ffdff0"/>
          <stop offset="100%" stop-color="#ffeec8"/>
        </linearGradient>
      </defs>
      <circle cx="48" cy="48" r="46" fill="url(#bg)"/>
      <circle cx="33" cy="40" r="5" fill="#8b6f5a"/>
      <circle cx="63" cy="40" r="5" fill="#8b6f5a"/>
      <path d="M30 57c5 8 31 8 36 0" fill="none" stroke="#8b6f5a" stroke-width="5" stroke-linecap="round"/>
      <circle cx="24" cy="52" r="6" fill="#ffc7da" opacity="0.85"/>
      <circle cx="72" cy="52" r="6" fill="#ffc7da" opacity="0.85"/>
    </svg>`
  );

const AVATAR_BY_ROLE: Record<string, string> = {
  principal_investigator: "/img/agent_role/principal_investigator.png",
  computational_biologist: "/img/agent_role/computational_biologist.png",
  machine_learning_specialist: "/img/agent_role/machine_learning_specialist.png",
  scientific_critic: "/img/agent_role/scientific_critic.png"
};

function roleDisplayName(roleId: string, isUser: boolean) {
  if (isUser) return "User";
  if (!roleId) return "Assistant";
  return roleId.split("_").join(" ").replace(/\b\w/g, (m: string) => m.toUpperCase());
}

function roleAvatar(roleId: string, isUser: boolean) {
  if (isUser) return USER_AVATAR;
  return AVATAR_BY_ROLE[roleId] || DEFAULT_AVATAR;
}

function renderMarkdown(text: string) {
  const html = marked.parse(text || "", { async: false }) as string;
  return DOMPurify.sanitize(html);
}

export function ChatTimeline({ items }: { items: ChatHistoryItem[] }) {
  return (
    <div className="chat-timeline">
      {items.length === 0 && (
        <div className="chat-empty">No messages yet. Start from the composer below.</div>
      )}
      {items.map((item, idx) => {
        const isUser = item.role === "user";
        const roleId = item.role_id || "";
        const roleLabel = roleDisplayName(roleId, isUser);
        const avatar = roleAvatar(roleId, isUser);
        return (
          <div key={idx} className={`chat-msg ${isUser ? "user" : "assistant"} with-avatar`}>
            <img
              className="chat-msg-avatar"
              src={avatar}
              alt={roleLabel}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).src = DEFAULT_AVATAR;
              }}
            />
            <div className="chat-msg-content">
              <div className="chat-msg-role">{roleLabel}</div>
              <div
                className="chat-msg-body"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(item.content || "") }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
