from openai import OpenAI

def gpt4_summary(text, openai_api_key, choose_pdf = None):
    """
    Utilise GPT-4 pour résumer le texte donné.
    :param text: Texte à résumer.
    :param openai_api_key: Clé API pour OpenAI.
    :return: Résumé du texte.
    """

    client =  OpenAI(api_key=openai_api_key)

    try:
        # if 0 in choose_pdf : 
        response0 = client.chat.completions.create(
        model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
            messages=[
                {
                    "role": "system", 
                    "content": "Crée un article d'apres les articles donne ci-dessous, au debut met le titre du sujet en deux mots, en expliquant les endroits qui sont le plus propice pour que cela fonctionne dans le monde, et le type d'endroit, et comment ? Assure-toi que le ton soit accessible et motivant. Ne t'arrête pas de développer les idées après une seule itération; si un concept peut être expliqué plus en détail, continue à le faire."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.7,  # Un peu plus élevé pour encourager la créativité
            max_tokens=4_000,  # Augmenter pour permettre une réponse plus longue
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )
        if 1 in choose_pdf : 
            response1 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": "Fais moi un mode d'emploi a partir des textes qui fais un resume de la partie, au debut met le titre du sujet en deux mots, de ce qu'il faut faire, et les logiciels, avec un lexique"},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
        if 2 in choose_pdf : 
            response2 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": "Imagine-toi être un journaliste qui veut faire un long article sur les sujets de textes ci-dessous. au debut met le titre du sujet en deux mots, Je voudrais que tu rédiges un synopsis en cinq grandes parties qui va servir ultérieurement à la rédaction du long article de ce journaliste. Ce synopsis doit inclure une partie sur ce qu'esu sujet et comment cela fonctionne. Le synopsis doit aussi inclure une partie sur les enjeux du sujet. ce  synopsis doit aussi inclure une partie sur le rôle des éditeurs de logiciels dans le domaine du sujet."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            syntaxe = response2.choices[0].message.content
            one_part = syntaxe[syntaxe.find("1."):]
            one_part = one_part.split('\n')[0]

            two_part = syntaxe[syntaxe.find("2."):]
            two_part = two_part.split('\n')[0]

            three_part = syntaxe[syntaxe.find("3."):]
            three_part = three_part.split('\n')[0]

            four_part = syntaxe[syntaxe.find("4."):]
            four_part = four_part.split('\n')[0]

            five_part = syntaxe[syntaxe.find("5."):]
            five_part = five_part.split('\n')[0]

            response3 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": f"Ecris cette partie a partir des articles donnes pour un profesionel {one_part}, avec des exemples et liens"},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            response4 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": f"Ecris cette partie a partir des articles donnes pour un profesionel {two_part}, avec des exemples et liens"},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            response5 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": f"Ecris cette partie a partir des articles donnes pour un profesionel {three_part}, avec des exemples et liens"},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            response6 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": f"Ecris cette partie a partir des articles donnes pour un profesionel {four_part}, avec des exemples et liens"},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            response7 = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Assurez-vous que "gpt-4" est le bon nom du modèle au moment de l'exécution
                messages=[
                    {"role": "system", "content": f"Ecris cette partie a partir des articles donnes pour un profesionel {five_part}, avec des exemples et liens"},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,  # Un peu plus élevé pour encourager la créativité
                max_tokens=1000,  # Augmenter pour permettre une réponse plus longue
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
        return(
                response0.choices[0].message.content, 
               response1.choices[0].message.content if 1 in choose_pdf else 0, 
                (
                    response2.choices[0].message.content + '\n', 
                    response3.choices[0].message.content + '\n',
                    response4.choices[0].message.content + '\n', 
                    response5.choices[0].message.content + '\n', 
                    response6.choices[0].message.content + '\n', 
                    response7.choices[0].message.content + '\n'
                ) if 2 in choose_pdf else 0
        )
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
