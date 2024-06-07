import React, { ReactNode, createContext, useContext, useState } from "react";
import { Video, PoseTrackingData } from "../types";

interface VideoContextType {
  currentTime: number; // in frames
  setCurrentTime: (time: number) => void;
  timeSetter: string;
  setTimeSetter: (timeSetter: string) => void;
  videoMeta: Video | undefined;
  setVideoMeta: (video: Video) => void;
  scale: Scale; // unit x scale = pixels
  setScale: (scale: Scale) => void;
  poseTrackingData: PoseTrackingData | undefined;
  setPoseTrackingData: (data: PoseTrackingData) => void;
  videoStatus: VideoStatus;
  setVideoStatus: (status: VideoStatus) => void;
}

type Scale = {
  ratio: number;
  unit: string;
};

type VideoStatus = {
  status: "playing" | "paused" | "ended";
  playbackRate: number;
  volume: number;
};

const VideoContext = createContext<VideoContextType | undefined>(undefined);

interface VideoProviderProps {
  children: ReactNode;
}

export const VideoProvider: React.FC<VideoProviderProps> = ({ children }) => {
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [timeSetter, setTimeSetter] = useState<string>("video");
  const [videoMeta, setVideoMeta] = useState<Video | undefined>(undefined);
  const [scale, setScale] = useState<Scale>({ ratio: 1, unit: "" });
  const [poseTrackingData, setPoseTrackingData] = useState<
    PoseTrackingData | undefined
  >(undefined);
  const [videoStatus, setVideoStatus] = useState<VideoStatus>({
    status: "paused",
    playbackRate: 1,
    volume: 100,
  });
  return (
    <VideoContext.Provider
      value={{
        currentTime,
        setCurrentTime,
        timeSetter,
        setTimeSetter,
        videoMeta,
        setVideoMeta,
        scale,
        setScale,
        poseTrackingData,
        setPoseTrackingData,
        videoStatus,
        setVideoStatus,
      }}
    >
      {children}
    </VideoContext.Provider>
  );
};

export function useVideo() {
  const context = useContext(VideoContext);
  if (context === undefined) {
    throw new Error("useVideo must be used within a VideoProvider");
  }
  return context;
}
