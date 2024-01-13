import axios from "axios";
import authHeader from "./auth-header";

class UserService {
  getPublicContent() {
    return axios.get("all");
  }

  getProfile() {
    console.log(authHeader());
    return axios.get("users/profile", { headers: authHeader() });
  }

  newJob(videoId: number, maxDet: number, flow: boolean, animal: string, segmModel: string, poseModel: string, flowModel: null | string) {
    console.log({ videoId, maxDet, flow, animal, segmModel, poseModel, flowModel });
    return axios.put(
      "jobs/",
      { videoId, maxDet, flow, animal, segmModel, poseModel, flowModel },
      { headers: authHeader() }
    );
  }

  cancelJob(jobId: string) {
    return axios.delete(`jobs/${jobId}`, { headers: authHeader() });
  }

  getJobs() {
    return axios.get("jobs/", { headers: authHeader() });
  }

  getModels() {
    return axios.get("models/", { headers: authHeader() });
  }

  getTrackResult(videoId: number) {
    return axios.get(`tracks/${videoId}`, { headers: authHeader() });
  }

  deleteVideo(videoId: number) {
    return axios.delete(`videos/${videoId}`, { headers: authHeader() });
  }
}

export default new UserService();
