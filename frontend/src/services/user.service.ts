import axios from "axios";
import authHeader from "./auth-header";

class UserService {
  getUploadVideoApi(): string {
    return axios.defaults.baseURL + "/videos/";
  }

  getUploadImageApi(datasetId: number): string {
    return axios.defaults.baseURL + `/datasets/${datasetId}/images`;
  }

  getUploadCOCOApi(datasetId: number): string {
    return axios.defaults.baseURL + `/datasets/${datasetId}/coco`;
  }

  getPublicContent() {
    return axios.get("all");
  }

  getProfile() {
    console.log(authHeader());
    return axios.get("users/profile", { headers: authHeader() });
  }

  newTrackJob(
    videoId: number,
    maxDet: number,
    flow: boolean,
    animal: string,
    segmModel: string,
    poseModel: string,
    flowModel: null | string,
    maxTrainingFrames: null | number,
    baseVideoId: null | number,
    enableStage1: boolean,
    enableStage2: boolean
  ) {
    console.log({
      videoId,
      maxDet,
      flow,
      animal,
      segmModel,
      poseModel,
      flowModel,
      maxTrainingFrames,
      baseVideoId,
      enableStage1,
      enableStage2,
    });
    return axios.put(
      "jobs/track",
      {
        videoId,
        maxDet,
        flow,
        animal,
        segmModel,
        poseModel,
        flowModel,
        maxTrainingFrames,
        baseVideoId,
        enableStage1,
        enableStage2,
      },
      { headers: authHeader() }
    );
  }

  newDetTrainJob(
    config: string,
    datasetId: number,
    valRatio: number,
    modelName: string
  ) {
    return axios.post(
      "jobs/det",
      { config, datasetId, valRatio, modelName },
      { headers: authHeader() }
    );
  }

  getDetTrainJobs() {
    return axios.get("jobs/det", { headers: authHeader() });
  }

  getPoseTrainJobs() {
    return axios.get("jobs/pose", { headers: authHeader() });
  }

  newPoseTrainJob(
    config: string,
    datasetId: number,
    valRatio: number,
    modelName: string,
    links: string[][],
    swaps: string[][]
  ) {
    return axios.post(
      "jobs/pose",
      { config, datasetId, valRatio, modelName, links, swaps },
      { headers: authHeader() }
    );
  }

  cancelJob(jobId: string) {
    return axios.delete(`jobs/${jobId}`, { headers: authHeader() });
  }

  getTrackJobs() {
    return axios.get("jobs/track", { headers: authHeader() });
  }

  getModels() {
    return axios.get("models/", { headers: authHeader() });
  }

  getModelById(modelId: string) {
    return axios.get(`models/${modelId}`, { headers: authHeader() });
  }

  deleteModel(modelId: string) {
    return axios.delete(`models/${modelId}`, { headers: authHeader() });
  }

  updateModel(modelId: string, trained: Boolean) {
    return axios.post(
      `models/${modelId}`,
      { trained },
      { headers: authHeader() }
    );
  }

  getTrackResult(videoId: number) {
    return axios.get(`tracks/${videoId}`, { headers: authHeader() });
  }

  deleteVideo(videoId: number) {
    return axios.delete(`videos/${videoId}`, { headers: authHeader() });
  }

  getDatasets() {
    return axios.get("datasets/", { headers: authHeader() });
  }

  getDatasetById(id: number) {
    return axios.get(`datasets/${id}`, { headers: authHeader() });
  }

  createNewDataset(name: string, keypoints: string[], animalName: string) {
    return axios.post(
      "datasets/",
      { name, keypoints, animalName },
      { headers: authHeader() }
    );
  }

  getImageList(datasetId: number) {
    return axios.get(`datasets/${datasetId}/images`, {
      headers: authHeader(),
    });
  }

  deleteDataset(id: number) {
    return axios.delete(`datasets/${id}`, { headers: authHeader() });
  }

  postAnnotations(dataset_id: number, imageId: number, annotations: any) {
    return axios.post(
      `datasets/${dataset_id}/images/${imageId}`,
      { annotations },
      { headers: authHeader() }
    );
  }

  getAnnotationsByImageId(dataset_id: number, imageId: number) {
    return axios.get(`datasets/${dataset_id}/images/${imageId}`, {
      headers: authHeader(),
    });
  }

  putNewInstance(dataset_id: number, imageId: number) {
    return axios.put(
      `datasets/${dataset_id}/images/${imageId}`,
      {},
      { headers: authHeader() }
    );
  }

  deleteInstance(dataset_id: number, imageId: number, annotation_id: number) {
    return axios.delete(
      `datasets/${dataset_id}/images/${imageId}/${annotation_id}`,
      { headers: authHeader() }
    );
  }

  getModelConfigs() {
    return axios.get("models/configs", { headers: authHeader() });
  }
}

export default new UserService();
