import { message } from "antd";
import { AxiosError } from "axios";

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
  switch (seed) {
    case 0:
      return "#e77c8d";
    case 1:
      return "#c69255";
    case 2:
      return "#98a255";
    case 3:
      return "#56ad74";
    case 4:
      return "#5aa9a2";
    case 5:
      return "#5ea5c5";
    case 6:
      return "#a291e1";
    case 7:
      return "#e274cf";
  }
  if (typeof seed === "string") seed = hashCode(seed);
  seed = seed * 99999;
  return "#" + Math.floor(Math.abs(Math.sin(seed) * 16777215)).toString(16);
};

export const downloadFile = (data: any, filename: string) => {
  const content = JSON.stringify(data, null, 2);
  const blob = new Blob([content], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

export const logError = (e: AxiosError) => {
  console.error(e);
  if (
    e.response &&
    e.response.data &&
    typeof e.response.data === "object" &&
    "message" in e.response.data
  ) {
    message.error(e.response.data.message as string, 5);
  } else {
    message.error(e.message, 5);
  }
};
