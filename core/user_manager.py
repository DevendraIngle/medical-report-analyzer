from datetime import datetime
import json
import os
import uuid
import streamlit as st

# Define the path for the user database file
USER_DB_PATH = "user_profiles.json"

class UserManager:
    def __init__(self):
        self.users = self._load_users()

    def _load_users(self):
        """Loads user profiles from the JSON file."""
        if os.path.exists(USER_DB_PATH):
            try:
                with open(USER_DB_PATH, "r", encoding='utf-8') as f:
                    users = json.load(f)
                    st.sidebar.success(f"Loaded {len(users)} user profiles.")
                    return users
            except json.JSONDecodeError:
                st.sidebar.error(f"Error decoding JSON from {USER_DB_PATH}. Starting with empty user list.")
                return []
            except Exception as e:
                st.sidebar.error(f"Error loading user profiles: {e}. Starting with empty user list.")
                return []
        else:
            st.sidebar.info("No user profile database found. Starting fresh.")
            return []

    def _save_users(self):
        """Saves current user profiles to the JSON file."""
        try:
            with open(USER_DB_PATH, "w", encoding='utf-8') as f:
                json.dump(self.users, f, indent=4)
            st.sidebar.success("User profiles saved.")
        except Exception as e:
            st.sidebar.error(f"Error saving user profiles: {e}")

    def add_user(self, name, age, gender):
        """Adds a new user profile and returns their unique ID."""
        user_id = str(uuid.uuid4())
        new_user = {
            "id": user_id,
            "name": name,
            "age": age,
            "gender": gender,
            "created_at": str(datetime.now()) # Optional: add creation timestamp
        }
        self.users.append(new_user)
        self._save_users()
        st.sidebar.success(f"New profile created for {name}.")
        return new_user

    def find_user(self, name, age, gender):
        """Finds an existing user by name, age, and gender."""
        for user in self.users:
            if user.get("name") == name and user.get("age") == age and user.get("gender") == gender:
                st.sidebar.info(f"Found existing profile for {name}.")
                return user
        st.sidebar.warning(f"No profile found for {name}.")
        return None

    def get_user_by_id(self, user_id):
        """Finds a user by their unique ID."""
        for user in self.users:
            if user.get("id") == user_id:
                return user
        return None

    def get_all_users(self):
        """Returns a list of all user profiles."""
        return self.users

    def delete_user(self, user_id):
        """Deletes a user profile by ID."""
        initial_count = len(self.users)
        self.users = [user for user in self.users if user.get("id") != user_id]
        if len(self.users) < initial_count:
            self._save_users()
            st.sidebar.success(f"Profile with ID {user_id} deleted.")
            return True
        else:
            st.sidebar.warning(f"Profile with ID {user_id} not found.")
            return False