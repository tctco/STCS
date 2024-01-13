export type User = {
  id: number;
  email: string;
  priority: number;
  trueName: string;
  department: string;
  bio: string;
  videos: Video[];
};

export type Video = {
  id: number;
  key: number; // same as id
  name: string;
  url: string;
  frameCnt: number;
  fps: number;
  width: number;
  height: number;
  timeUploaded: string;
  analyzed: boolean;
  size: number;
};

export type Task = {
  taskId: string;
  trueName: string;
  maxDet: number;
  videoName: string;
  created: string;
  started: string | null;
  ended: string | null;
  status: string;
  priority: number;
  totalFrames: number;
  owned: boolean;
};

export type TrackResult = {
  headers: {
    connections: number[][];
    interval: number[];
  };
  data: number[][][][]; // track x frame x kpts x (x,y)
};

export type Model = {
  id: string;
  config:string;
  checkpoint:string;
  animal:string;
  type:string;
  method:string;
}