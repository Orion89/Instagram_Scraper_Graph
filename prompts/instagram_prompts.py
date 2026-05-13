# Plantillas de prompts para el agente de browser-use


def get_instagram_scrape_prompt(
    hashtag: str, n_results: int, user_email: str, user_password: str
) -> str:
    """
    Genera el prompt para extraer posts de Instagram basados en un hashtag.
    """
    return f"""
Go to https://www.instagram.com/explore/search/keyword/?q=%23{hashtag}
The website corresponds to the search results for the hashtag #{hashtag} on Instagram.

If a login message appears or you are redirected to login, look for the login button and click it to log in.
The login credentials are:
email: {user_email}
password: {user_password}
Log in and then return to the search results for #{hashtag}.

Process:
1. Access at least {n_results} unique posts from the results list available at the link. You must distinguish posts by their URLs (corresponds to the `post_link` data extracted from each post) to identify them as unique. You must scroll through the results list as needed, using scroll action.
2. You must click on each post in the results list to access it. Once you have extracted the data from a post, you must return to the initial results list available at https://www.instagram.com/explore/search/keyword/?q=%23{hashtag} and repeat the data extraction process with the next post by clicking on it, and so on with each post in the initial results list.
3. For each post from the results list available at the link, extract exactly:
   - user_name: The name of the account that made the post.
   - post_hashtags: A list of all hashtags found in the post (e.g., ["#tech", "#ai"]). You may need to scroll through the text of a particular post to see the hashtags if the post is too long.
   - likes_count: The number of likes (as an integer).
   - post_link: The direct URL to the post.
   - post_caption: The full text description/caption of the post.
   - image_description: A brief description of the visual content (alt text or what is seen).
   - comments_count: The number of comments on the post, if available. It's the number next to the speech bubble icon at the bottom of each post's image.
   - repost_count: The number of reposts, if available. It's the number next to the looping arrow icon at the bottom of each post's image.

You must scroll through the results list as needed, using scroll action, to reach the target of {n_results} posts. You must distinguish posts by their URLs to identify them as unique (corresponds to the `post_link` data extracted from each post).
Remember that once you have accessed a post from the results list available at the link and extracted its data, you must return to the initial list of search results and repeat the data extraction process with the next post by clicking on it, and so on with each post in the initial results list.
The posts you access should only come from the initial results list.
If you need to click on posts to see the details, do so and then return to the list.
""".strip()
