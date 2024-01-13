import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { setMessage } from "./message";

import authService from "../services/auth.service";

const user = authService.getCurrentUser()

export const register = createAsyncThunk(
  "auth/register",
  async({email, password, trueName, department, bio}, thunkAPI) => {
    try {
      const response = await authService.register(email, password, trueName, department, bio);
      return response.data;
    } catch (e) {
      const message = (e.response && e.response.data && e.response.data.message) || e.message || e.toString();
      thunkAPI.dispatch(setMessage(message));
      return thunkAPI.rejectWithValue({ message });
    }
  }
)

export const login = createAsyncThunk(
  "auth/login",
  async({email, password}, thunkAPI) => {
    try {
      const response = await authService.login(email, password);
      return response.data;
    } catch (e) {
      const message = (e.response && e.response.data && e.response.data.message) || e.message || e.toString();
      thunkAPI.dispatch(setMessage(message));
      return thunkAPI.rejectWithValue({ message });
    }
  }
)

export const logout = createAsyncThunk(
  "auth/logout",
  async () => {
    console.log('auth/logout')
    await authService.logout();
  }
)

const initialState = user ? {isLoggedIn: true, user} : {isLoggedIn: false, user: null};
console.log('initialState', initialState)

const authSlice = createSlice({
  name: "auth",
  initialState,
  extraReducers: {
    [register.fulfilled]: (state, action) => {
      console.log('register.fulfilled')
      state.isLoggedIn = false;
    },
    [register.rejected]: (state, action) => {
      console.log("register.rejected")
      state.isLoggedIn = false;
    },
    [login.fulfilled]: (state, action) => {
      console.log('login.fulfilled')
      state.isLoggedIn = true;
      state.user = action.payload;
    },
    [login.rejected]: (state, action) => {
      console.log("login.rejected")
      state.isLoggedIn = false;
      state.user = null;
    },
    [logout.fulfilled]: (state, action) => {
      console.log("logout.fulfilled");
      state.isLoggedIn = false;
      state.user = null;
    }
  }
})

const { reducer } = authSlice;
export default reducer;