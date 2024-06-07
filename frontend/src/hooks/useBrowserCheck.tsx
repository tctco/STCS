import { useEffect, useState } from "react";
const useBrowserCheck = () => {
  const [isChrome, setIsChrome] = useState(true);

  useEffect(() => {
    setIsChrome(navigator.userAgent.indexOf("Chrome") !== -1);
  }, []);

  return isChrome;
};

export default useBrowserCheck;
