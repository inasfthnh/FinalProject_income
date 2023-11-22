# tempat running aplikasi
import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#6F8FAF;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Citizens Income Prediction App </h1>
		    <h2 style="color:white;text-align:center;">Census Bureau </h2>
	    </div>
            """

desc_temp = """
            ### Citizens Income Prediction App
            This app will be used by the government to predict whether the personal income would be over 50K or not.
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
                        <img src="https://stickersmag.com/wp-content/uploads/2018/03/50.png">
                    </div>
                    """
                 )
        st.subheader('Final Project Fantastic Four')
        st.write("---")
        choice = st.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.write("---")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
