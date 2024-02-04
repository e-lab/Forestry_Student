import datetime
import streamlit as st
import extra_streamlit_components as stx

st.write("# Cookie Manager") 

cookie_manager = stx.CookieManager()

st.subheader("All Cookies:")
cookies = cookie_manager.get_all()
st.write(cookies)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Get Cookie:")
    cookie = st.text_input("Cookie", key="0")
    clicked = st.button("Get")
    if clicked:
        value = cookie_manager.get(cookie="messages")
        st.write(value)
with c2:
    st.subheader("Set Cookie:")
    cookie = st.text_input("Cookie", key="1")
    val = st.text_input("Value")
    val = [{"role": "assistant", "content": "Hello! The name's euGenio. I'm here to help you with your pipeline. Ask me a question!"}]
    if st.button("Add"):
        cookie_manager.set(cookie, val) # Expires in a day by default
with c3:
    st.subheader("Delete Cookie:")
    cookie = st.text_input("Cookie", key="2")
    if st.button("Delete"):
        cookie_manager.delete(cookie)
