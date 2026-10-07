import os
import requests
import streamlit as st

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CityAI Assistant",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #172554 100%
        );
        color: white;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: white;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Cards */
    .info-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 15px;
    }

    .tool-card {
        padding: 15px;
        border-radius: 14px;
        background: rgba(59,130,246,0.10);
        border: 1px solid rgba(96,165,250,0.20);
        margin-top: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 15px;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "approved" not in st.session_state:
    st.session_state.approved = False


# =========================================================
# WEATHER TOOL
# =========================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key is missing."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather')}"

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return (
            f"Weather in {city}:\n"
            f"Condition: {description}\n"
            f"Temperature: {temp}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%"
        )

    except requests.RequestException as e:
        return f"Weather API error: {str(e)}"


# =========================================================
# NEWS TOOL
# =========================================================

if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None


@tool
def get_news(city: str) -> str:
    """Get the latest news about a city."""

    if not tavily_client:
        return "Tavily API key is missing."

    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return f"No latest news found for {city}."

        news_list = []

        for result in results:
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")

            news_list.append(
                f"### {title}\n"
                f"{content[:300]}...\n"
                f"Source: {url}"
            )

        return (
            f"Latest news in {city}:\n\n"
            + "\n\n".join(news_list)
        )

    except Exception as e:
        return f"News API error: {str(e)}"


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def get_llm():

    return ChatMistralAI(
        model="codestral-2508",
        temperature=0
    )


# =========================================================
# AGENT
# =========================================================

@st.cache_resource
def get_agent():

    llm = get_llm()

    agent = create_agent(
        llm,
        tools=[
            get_weather,
            get_news
        ],
        system_prompt="""
        You are CityAI, a helpful city assistant.

        You can:
        1. Get current weather information.
        2. Get latest city news.

        Use tools whenever the user asks for current weather
        or latest news.

        Give clear, concise and friendly answers.
        Do not invent weather or news information.
        """,
    )

    return agent


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌆 CityAI")

    st.markdown(
        """
        <div class="info-card">
        <b>AI City Assistant</b><br><br>
        Ask about:
        <br>🌤️ Current weather
        <br>📰 Latest city news
        <br>🏙️ City information
        <br>🤖 AI-powered answers
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🛠️ Available Tools")

    st.success("🌤️ Weather")
    st.info("📰 Latest News")

    st.divider()

    st.markdown("### 🔐 API Status")

    if OPENWEATHER_API_KEY:
        st.success("OpenWeather Connected")
    else:
        st.error("OpenWeather Missing")

    if TAVILY_API_KEY:
        st.success("Tavily Connected")
    else:
        st.error("Tavily Missing")

    if MISTRAL_API_KEY:
        st.success("Mistral Connected")
    else:
        st.warning(
            "Mistral key may be loaded automatically "
            "by LangChain environment configuration."
        )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.approved = False

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌆 CityAI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI-powered assistant for weather, news and city insights'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# WELCOME SCREEN
# =========================================================

if len(st.session_state.messages) == 0:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
            <h3>🌤️ Weather</h3>
            <p>
            Get current temperature, humidity,
            conditions and more.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
            <h3>📰 Latest News</h3>
            <p>
            Search the latest news and updates
            about any Indian city.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
            <h3>🤖 AI Agent</h3>
            <p>
            Mistral automatically decides
            which tool should be used.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 💡 Try asking")

    examples = [
        "What is the weather in Delhi?",
        "Give me latest news about Mumbai",
        "What's the weather in Bangalore?",
        "Tell me the latest news from Kolkata"
    ]

    for example in examples:
        if st.button(
            f"💬 {example}",
            use_container_width=True
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": example
                }
            )

            st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask about weather, news or a city..."
)


# =========================================================
# PROCESS USER INPUT
# =========================================================

if user_input:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("🤖 CityAI is thinking..."):

            try:

                # =================================================
                # 🌦️ WEATHER REQUEST
                # =================================================
                # Do not depend on Codestral's tool selection for
                # weather. Route obvious weather requests directly
                # to the existing get_weather tool.
                # =================================================

                weather_keywords = [
                    "weather",
                    "temperature",
                    "temp",
                    "rain",
                    "humidity",
                    "forecast"
                ]

                text_lower = user_input.lower()

                is_weather_request = any(
                    word in text_lower
                    for word in weather_keywords
                )

                if is_weather_request:

                    llm = get_llm()

                    # Extract city with Codestral-2508
                    city_result = llm.invoke(
                        f"""
Extract only the city name from this request.

Request:
{user_input}

Return ONLY the city name.
Do not return any explanation.
"""
                    )

                    city = city_result.content.strip()
                    city = city.replace('"', "").replace("'", "")
                    city = city.rstrip(".,?! ")

                    if not city:

                        final_response = (
                            "Please mention the city name, "
                            "for example: What is the weather in Delhi?"
                        )

                    else:

                        # Call the real OpenWeather tool directly
                        weather_result = get_weather.invoke(
                            {
                                "city": city
                            }
                        )

                        # Ask Codestral-2508 to present the result
                        answer_result = llm.invoke(
                            f"""
You are CityAI, a helpful city assistant.

User asked:
{user_input}

Weather tool result:
{weather_result}

Give a short, natural answer to the user.

Rules:
- Use only the information returned by the weather tool.
- Do not invent weather information.
- Do not say that you don't have access to tools.
- If the tool returned an error, clearly explain that error.
"""
                        )

                        final_response = answer_result.content

                # =================================================
                # 📰 OTHER REQUESTS → AGENT
                # =================================================

                else:

                    agent = get_agent()

                    result = agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ]
                        }
                    )

                    messages = result.get(
                        "messages",
                        []
                    )

                    final_response = None

                    for message in reversed(messages):

                        if (
                            message.__class__.__name__
                            == "AIMessage"
                            and message.content
                        ):
                            final_response = message.content
                            break

                    if not final_response:

                        final_response = (
                            "Sorry, I couldn't generate a response."
                        )

                # =================================================
                # DISPLAY RESPONSE
                # =================================================

                st.markdown(final_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response
                    }
                )

            except Exception as e:

                error_message = f"⚠️ Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )
