import axios from "axios";

class AuthService {
  async login(email: string, password: string) {
    return axios
      .post("users/login", {
        email,
        password,
      })
      .then((response) => {
        if (response.data.accessToken) {
          localStorage.setItem("user", JSON.stringify(response.data));
        }
        console.log(response);

        return response.data;
      });
  }

  logout() {
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
