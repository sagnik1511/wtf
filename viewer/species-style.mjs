// Presentation only: species identifiers remain arbitrary configuration keys.
export function speciesStyles(species) {
  const roles=new Set();let extraIndex=2;
  return new Map(Object.entries(species).map(([name, spec]) => {
    const index=roles.has(spec.diet)?extraIndex++:spec.diet==='prey'?1:0;
    roles.add(spec.diet);
    const hue=(index*137.508+25)%360;
    const defaults=index===0?['#7de2bd','#60a5fa']:index===1?['#ff9a77','#f6d365']:
      [`hsl(${hue.toFixed(2)}, 72%, 72%)`,`hsl(${((hue+45)%360).toFixed(2)}, 76%, 60%)`];
    const female=spec.display?.color??defaults[0];
    const male=spec.display?.male_color??defaults[1];
    return [name,{name, female, male, color:female, symbol:spec.diet==='prey'?'▲':'●'}];
  }));
}
