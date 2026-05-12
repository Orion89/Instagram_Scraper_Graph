# Plantillas de prompts para el agente de browser-use

def get_instagram_scrape_prompt(hashtag: str, n_results: int, user_email: str, user_password: str) -> str:
    """
    Genera el prompt para extraer posts de Instagram basados en un hashtag.
    """
    return f"""
Go to https://www.instagram.com/explore/search/keyword/?q=%23{hashtag}
The website corresponds to the search results for the hashtag #{hashtag} on Instagram.

If a login message appears or you are redirected to login:
The login credentials are:
email: {user_email}
password: {user_password}
Log in and then return to the search results for #{hashtag}.

Process:
1. Access at least {n_results} unique posts from the results list. 
2. For each post, extract exactly:
   - user_name: The name of the account that made the post.
   - post_hashtags: A list of all hashtags found in the post (e.g., ["#tech", "#ai"]).
   - likes_count: The number of likes (as an integer).
   - post_link: The direct URL to the post.
   - post_caption: The full text description/caption of the post.
   - image_description: A brief description of the visual content (alt text or what is seen).

You must scroll through the results list as needed to reach the target of {n_results} posts.
If you need to click on posts to see the details, do so and then return to the list.
""".strip()
