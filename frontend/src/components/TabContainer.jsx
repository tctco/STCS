import React from "react";
import styles from "./TabContainer.module.css";

export default class TabContainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = { active: false, observing: false };
    this.container = React.createRef();
  }

  componentDidMount() {
    if (this.props.setScale) {
      this.setState({ observing: true });
      this.observer = new ResizeObserver(() => {
        this.props.setScale({
          title: this.props.title,
          width: this.container.current.offsetWidth,
          height: this.container.current.offsetHeight,
        });
      });
      this.observer.observe(this.container.current);
    }
  }

  componentWillUnmount() {
    if (this.state.observing) this.observer.disconnect();
  }

  render() {
    let hover = this.state.active
      ? { boxShadow: "0 7px 14px rgba(0,0,0,0.25), 0 5px 5px rgba(0,0,0,0.22)" }
      : {};

    return (
      <div
        style={{ ...this.props.style, margin: "5px 20px 20px 10px" }}
        ref={this.container}
      >
        <div
          className={styles.tag}
          style={hover}
          onMouseOver={() => this.setState({ active: true })}
          onMouseLeave={() => this.setState({ active: false })}
        >
          <h6 style={{ cursor: "default" }}>{this.props.title}</h6>
        </div>
        <div
          className={styles.container}
          style={hover}
          onMouseOver={() => this.setState({ active: true })}
          onMouseLeave={() => this.setState({ active: false })}
        >
          <div style={{ position: "relative" }}>{this.props.children}</div>
        </div>
      </div>
    );
  }
}
