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

export enum JobStatus {
  queued = "queued",
  started = "started",
  deferred = "deferred",
  finished = "finished",
  stopped = "stopped",
  scheduled = "scheduled",
  failed = "failed",
  canceled = "canceled",
}

export type Job = {
  taskId: string;
  trueName: string;
  maxDet: number;
  videoName: string;
  created: string;
  started: string | null;
  ended: string | null;
  status: JobStatus;
  priority: number;
  totalFrames: number;
  owned: boolean;
  progress: number;
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
  config: string;
  checkpoint: string;
  animal: string;
  type: string;
  method: string;
  name: string;
  owned: boolean;
  trained: boolean;
};

type IntervalData = {
  intervals: number[];
  rawTrackID: number;
  trackID: number | null;
};

export type PoseTrackingData = {
  headers: {
    connections: number[][];
    interval: number[]; // [start, end frame]
    tracklets: IntervalData[];
  };
  data: number[][][][]; // individual x frame x kpts x xy
};

export type Annotation = {
  id: number | null;
  image_id: number;
  keypoints: number[]; // [x, y, v, x, y, v]
  area: number;
  polygon: number[][]; //[[x, y, x, y...], []]
};
