export default function authHeader() {
  if (localStorage.getItem("user")) {
    return {
      Authorization:
        "Bearer " + JSON.parse(localStorage.getItem("user")).accessToken,
    };
  } else if (sessionStorage.getItem("user")) {
    return {
      Authorization:
        "Bearer " + JSON.parse(sessionStorage.getItem("user")).accessToken,
    };
  } else {
    return {};
  }
}
