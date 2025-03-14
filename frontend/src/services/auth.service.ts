import axios from "axios";

class AuthService {
  async login(email: string, password: string, rememberMe: boolean = false) {
    console.log("authService.login", email, password, rememberMe);
    return axios
      .post("users/login", {
        email,
        password,
      })
      .then((response) => {
        if (response.data.accessToken) {
          if (rememberMe) {
            localStorage.setItem("user", JSON.stringify(response.data));
          } else {
            sessionStorage.setItem("user", JSON.stringify(response.data));
          }
        }
        console.log(response);

        return response.data;
      });
  }

  checkExpiration() {
    const user = localStorage.getItem("user");
    if (user) {
      const { exp } = JSON.parse(user).accessToken;
      if (Date.now() >= exp * 1000) {
        localStorage.removeItem("user");
        sessionStorage.removeItem("user");
      }
    }
  }

  logout() {
    sessionStorage.removeItem("user");
    localStorage.removeItem("user");
  }

  register(
    email: string,
    password: string,
    trueName: string,
    department: string,
    bio: string
  ) {
    return axios.post("users/register", {
      email,
      password,
      trueName,
      department,
      bio,
    });
  }

  getCurrentUser() {
    const user = localStorage.getItem("user");
    if (user) {
      return JSON.parse(user);
    }
    return null;
  }
}

export default new AuthService();
