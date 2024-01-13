const hashCode = (str: string) => {
  let hash = 0,
    i: number,
    chr: number;
  if (str.length === 0) return hash;
  for (i = 0; i < str.length; i++) {
    chr = str.charCodeAt(i);
    hash = (hash << 5) - hash + chr;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
};

export const randomColor = (seed: number | string) => {
  if (typeof seed === "string") seed = hashCode(seed);
  return "#" + Math.floor(Math.abs(Math.sin(seed) * 16777215)).toString(16);
};
