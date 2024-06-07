const Resizer = (props) => {
  return (
    <div style={{ ...props.style }}>
      <div
        style={{
          resize: "horizontal",
          overflow: "auto",
          display: "inline-block",
          minWidth: props.minWidth,
          width: "100%",
          maxWidth: "100%",
        }}
      >
        {props.children}
      </div>
    </div>
  );
};

export default Resizer;
