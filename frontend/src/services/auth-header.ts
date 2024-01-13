export default function authHeader() {
  const user = localStorage.getItem("user");

  if (user && JSON.parse(user).accessToken) {
    return { Authorization: "Bearer " + JSON.parse(user).accessToken };
  } else {
    return {};
  }
}
