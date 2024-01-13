import { Carousel } from "antd";
import React from "react";

const contentStyle: React.CSSProperties = {
  margin: 0,
  height: "90vh",
  color: "#fff",
  lineHeight: "90vh",
  textAlign: "center",
  background: "#364d79",
};

const WelcomePage = () => {
  const onChange = (currentSlide: number) => {
    console.log(currentSlide);
  };
  return (
    <Carousel afterChange={onChange}>
      <div>
        <h2 style={contentStyle}>Welcome to segTracker.ai!</h2>
      </div>
      <div>
        <h2 style={contentStyle}>Track animals with ease</h2>
      </div>
      <div>
        <h2 style={contentStyle}>Ensembled deep learning</h2>
      </div>
    </Carousel>
  );
};

export default WelcomePage;
