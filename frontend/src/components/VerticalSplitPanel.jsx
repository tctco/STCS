import React from "react";
import styles from "./VerticalSplitPanel.module.css";
import { Empty } from "antd";

export default class VerticalSplitPanel extends React.Component {
  constructor(props) {
    super(props);
    this.separator = React.createRef();
    this.left = React.createRef();
    this.right = React.createRef();
    this.container = React.createRef();
    this.state = {
      separatorLeft: null,
      leftWidth: null,
      rightWidth: null,
    };
    this.handleMouseDown = this.handleMouseDown.bind(this);
  }

  handleMouseDown(event) {
    let startX = event.clientX;
    let startY = event.clientY;
    let containerWidth = this.container.current.offsetWidth;
    let startLeftWidth = this.left.current.offsetWidth;
    let startRightWidth = this.right.current.offsetWidth;
    let startSeparatorOffsetLeft = this.separator.current.offsetLeft;

    let handleMouseMove = (e) => {
      let delta = {
        x: e.clientX - startX,
        y: e.clientY - startY,
      };
      delta.x = Math.min(
        Math.max(delta.x, -startLeftWidth + 10),
        startRightWidth - 10
      );
      this.setState({
        separatorLeft:
          ((startSeparatorOffsetLeft + delta.x) / containerWidth) * 100,
        leftWidth: ((startLeftWidth + delta.x) / containerWidth) * 100,
        rightWidth: ((startRightWidth - delta.x) / containerWidth) * 100,
      });
    };

    document.addEventListener("mousemove", handleMouseMove);

    let handleMouseUp = () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mouseup", handleMouseUp);
  }

  componentDidMount() {
    this.setState({
      drawBoardHeight: this.left.current.offsetHeight,
    });
  }

  render() {
    return (
      <div className={styles.container} ref={this.container}>
        <div
          className={styles.left}
          ref={this.left}
          style={{ width: `${this.state.leftWidth}%` }}
        >
          {this.props.left}
        </div>
        <div
          title="drag me"
          className={styles.separator}
          onMouseDown={this.handleMouseDown}
          ref={this.separator}
          style={{ left: `${this.state.separatorLeft}%` }}
        ></div>
        <div
          className={styles.right}
          ref={this.right}
          style={{ width: `${this.state.rightWidth}%` }}
        >
          {this.props.right}
        </div>
      </div>
    );
  }
}

VerticalSplitPanel.defaultProps = {
  left: <Empty />,
  right: <Empty />,
};
