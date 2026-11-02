export type ParameterField = { type?: string; anyOf?: ParameterField[]; title?: string; default?: unknown; enum?: string[]; choices_by?: {field:string;options:Record<string,string[]>}; minimum?: number; maximum?: number; items?: ParameterField };
export type ParameterSchema = { properties: Record<string, ParameterField> };

export function parameterChoices(field:ParameterField,schema:ParameterSchema,values:Record<string,unknown>) {
  const dependency=field.choices_by;
  return dependency ? dependency.options[String(values[dependency.field] ?? schema.properties[dependency.field]?.default)] || [] : field.enum;
}

export function updateParameter(schema:ParameterSchema,values:Record<string,unknown>,key:string,value:unknown) {
  const next={...values,[key]:value};
  for (const [name,field] of Object.entries(schema.properties)) {
    if (field.choices_by?.field!==key) continue;
    const choices=parameterChoices(field,schema,next) || [];
    if (!choices.includes(String(next[name] ?? field.default))) next[name]=choices.includes(String(field.default)) ? field.default : choices[0];
  }
  return next;
}

export function parseArrayParameter(text: string, numeric: boolean): string[] | number[] {
  const values = text.trim() ? text.trim().split(/[,，\s]+/).filter(Boolean) : [];
  if (!numeric) return values;
  const numbers = values.map(Number);
  if (numbers.some(v => !Number.isFinite(v))) throw new Error('Use numbers separated by commas');
  return numbers;
}
