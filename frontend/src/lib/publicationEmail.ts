/** Publication requests leave through the user's email client, never a platform inbox. */
export function publicationEmail(name: string, reference: string): string {
  const body = [
    'Model / method: ' + name,
    'Research / model ID: ' + reference,
    '',
    'Author name:',
    'Author email (to display on the published model):',
    'Institution:',
    'Purpose and applicability:',
    'Code, weights and test report (attach ZIP or provide a download link):',
    'Data provenance and permission to share:',
    '',
    'Please review this model for publication on VenusFactory.',
  ].join('\n');
  return 'mailto:tanyang.august@sjtu.edu.cn?subject=' + encodeURIComponent('VenusFactory model publication request') + '&body=' + encodeURIComponent(body);
}
