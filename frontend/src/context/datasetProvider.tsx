import React, { useState, useContext } from "react";
export type Dataset = {
  id: number;
  name: string;
  keypoints: string[];
  animalName: string;
  userId: number;
  created: string;
  images: number;
  annotations: number;
};

interface DatasetContextType {
  dataset: Dataset;
  setDataset: (dataset: Dataset) => void;
}

const DatasetContext = React.createContext<DatasetContextType | undefined>(
  undefined
);

interface DatasetProviderProps {
  children: React.ReactNode;
}

export const DatasetProvider: React.FC<DatasetProviderProps> = ({
  children,
}) => {
  const [dataset, setDataset] = useState<Dataset | undefined>(undefined);

  return (
    <DatasetContext.Provider value={{ dataset, setDataset }}>
      {children}
    </DatasetContext.Provider>
  );
};

export const useDataset = () => {
  const context = useContext(DatasetContext);
  if (!context) {
    throw new Error("useJob must be used within a JobProvider");
  }
  return context;
};
