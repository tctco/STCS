import { ButtonProps, PopconfirmProps, Popconfirm, Button } from "antd";

type ConfirmButtonProps = {
  props?: ButtonProps;
  onConfirm?: (e: React.MouseEvent<HTMLElement, MouseEvent>) => void;
  onCancel?: (e: React.MouseEvent<HTMLElement, MouseEvent>) => void;
  description?: React.ReactNode;
  title?: string;
};

const ButtonWithConfirmation: React.FC<ConfirmButtonProps> = ({
  props,
  onConfirm,
  onCancel,
  description,
  title = "Delete",
}) => {
  const confirm: PopconfirmProps["onConfirm"] = (e) => {
    if (onConfirm) onConfirm(e);
  };

  const cancel: PopconfirmProps["onCancel"] = (e) => {
    if (onCancel) onCancel(e);
  };
  return (
    <Popconfirm
      title="Delete this dataset"
      description={description}
      onConfirm={confirm}
      onCancel={cancel}
    >
      <Button {...props} danger type="link">
        {title}
      </Button>
    </Popconfirm>
  );
};

export default ButtonWithConfirmation;
