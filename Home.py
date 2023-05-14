import streamlit as st
from PIL import Image
header = st.container()
image_container = st.container()
#col1, col2 = st.columns(2)
#col1, col2 = st.columns(2)


with header:
    st.header("Interpreting and Diagnosing Medical Images")

    # Define the allowed user roles
    ALLOWED_ROLES = ["Radiologist","Doctor","Technician","Administrator"]

    def login():
        st.title("Login")

        role = st.selectbox("Select your role: Radiologist, Doctor, Technician, Administrator", ALLOWED_ROLES)

        if role == "Administrator":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                # Validate system admin credentials
                if username == "admin" and password == "admin123":
                    st.success("Login successful as System Admin")
                    # Add your system admin logic here
                else:
                    st.error("Invalid credentials")

        elif role == "Radiologist":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                # Validate radiologist credentials
                if username == "radiologist" and password == "radiologist123":
                    st.success("Login successful as Radiologist")
                    # Add your radiologist logic here
                else:
                    st.error("Invalid credentials")

        elif role == "Technician":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                # Validate radiologist credentials
                if username == "technician" and password == "technician123":
                    st.success("Login successful as Technician")
                    # Add your radiologist logic here
                else:
                    st.error("Invalid credentials")

        elif role == "Doctor":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                # Validate doctor credentials
                if username == "doctor" and password == "doctor123":
                    st.success("Login successful as Doctor")
                    # Add your doctor logic here
                else:
                    st.error("Invalid credentials")

    # Run the login function
    login()

with image_container:
    image = Image.open("modalities.png")
    # Display the image with reduced size
    col1, col2, col3 = st.columns([1, 3, 1])  # Adjust the widths as desired
    col1.write("-----")
    col2.image(image)
    col3.write("-----")



