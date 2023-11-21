# tempat running aplikasi
import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#808080;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Citizen Income Prediction App </h1>
		    <h2 style="color:white;text-align:center;">Census Bureau </h2>
		    </div>
            """

desc_temp = """
            ### Citizen Income Prediction App
            This app will be used for government to predict whether the citizen get income above or below 50K
            #### Data Source
            - https://www.kaggle.com/datasets/tawfikelmetwally/census-income-dataset
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """


def main():

    stc.html(html_temp)

    menu = ['Home', 'Machine Learning']
    with st.sidebar:
        stc.html("""
                    <style>
                        .circle-image {
                            width: 130px;
                            height: 130px;
                            border-radius: 50%;
                            overflow: hidden;
                            box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
                        }
                        
                        .circle-image img {
                            width: 100%;
                            height: 100%;
                            object-fit: cover;
                        }
                    </style>
                    <div class="circle-image">
                        <img src="https://i.pinimg.com/originals/53/f0/38/53f03878e5e33cd473a02ab1af0064e1.jpg">
                    </div>
                    """
                 )
        st.write('Final Project Fantastic Four')
        choice = st.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
