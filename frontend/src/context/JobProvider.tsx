import React, { useContext, useState, useEffect } from "react";
import { Job } from "../types";
import userService from "../services/user.service";
import { logError } from "../utils";

interface JobContextType {
  selectedJob: Job | undefined;
  setSelectedJob: (job: Job) => void;
  jobs: Job[];
  setJobs: (jobs: Job[]) => void;
  requestJobs: () => void;
}

const JobContext = React.createContext<JobContextType | undefined>(undefined);

interface JobProviderProps {
  children: React.ReactNode;
}

export const JobProvider: React.FC<JobProviderProps> = ({ children }) => {
  const [selectedJob, setSelectedJob] = useState<Job | undefined>(undefined);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout>(null); // For polling jobs
  const requestJobs = () => {
    userService
      .getTrackJobs()
      .then((res) => {
        console.log("tracking jobs", res.data);
        setJobs(res.data as Job[]);
        clearTimeout(timeoutId);
        setTimeoutId(setTimeout(requestJobs, 1000 * 10));
      })
      .catch(logError);
  };

  useEffect(() => {
    requestJobs();
    return () => {
      clearTimeout(timeoutId);
    };
  }, []);

  return (
    <JobContext.Provider
      value={{ selectedJob, setSelectedJob, jobs, setJobs, requestJobs }}
    >
      {children}
    </JobContext.Provider>
  );
};

export const useJob = () => {
  const context = useContext(JobContext);
  if (!context) {
    throw new Error("useJob must be used within a JobProvider");
  }
  return context;
};

export default JobProvider;
