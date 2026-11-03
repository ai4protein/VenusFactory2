export type AllowanceEmail = {
  name: string;
  institution: string;
  email: string;
  contact: string;
  feedback: string;
  conversations: number;
  predictions: number;
};

export function allowanceEmailLink(request: AllowanceEmail): string {
  const subject = 'VenusFactory extra allowance (Feedback)';
  const body = [
    'Extra allowance request',
    '',
    `Name: ${request.name.trim()}`,
    `Institution: ${request.institution.trim()}`,
    `Email: ${request.email.trim()}`,
    `Contact: ${request.contact.trim()}`,
    '',
    `Extra conversations per day: ${request.conversations}`,
    `Extra model predictions per day: ${request.predictions}`,
    '',
    'Bugs or suggestions:',
    request.feedback.trim(),
  ].join('\n');
  return `mailto:tanyang.august@sjtu.edu.cn?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}
