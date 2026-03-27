import os
import time
import json
import base64
import asyncio
import logging
import requests
import jwt
import ffmpeg
import shutil
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Toggle features
USE_ELEVENLABS = False  # Set to True to enable ElevenLabs audio generation

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
KLING_AK = os.getenv("KLING_AK")
KLING_SK = os.getenv("KLING_SK")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ipqsbqjitkwhdfdixztq.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

class ContentEngineOutput(BaseModel):
    image_prompt: str
    video_prompt: str
    voiceover_script: str
    social_caption: str

def get_system_instruction(client_context: str) -> str:
    return f"""
    {client_context}
    
    Based on the user's workflow request, generate the image prompt, video prompt, voiceover script, and social media caption.
    Ensure that the final output tightly adheres to the requested JSON schema constraints. Ensure the image_prompt utilizes the brand's visual identity.
    """

def choose_client() -> tuple[str, str]:
    """Provides a CLI menu to select which client context to use from the 'clients' folder."""
    clients_dir = os.path.join(os.getcwd(), 'clients')
    os.makedirs(clients_dir, exist_ok=True)
    
    # Check for client files
    client_files = [f for f in os.listdir(clients_dir) if f.endswith('.txt')]
    
    if not client_files:
        print("\n[!] Inga kontext-filer hittades i mappen 'clients/'. Skapar en test-profil för Zaitex...")
        default_profile = os.path.join(clients_dir, 'zaitex.txt')
        with open(default_profile, 'w', encoding='utf-8') as f:
            f.write('You are the AI Content Engine for "Zaitex Solutions", a B2B agency helping companies scale and grow through social media and business tools.\nYour tone of voice is professional, modern, and engaging.')
        client_files = ['zaitex.txt']
    
    print("\n==================================")
    print(" VÄLJ KUND / BRAND KONTEXT")
    print("==================================")
    for i, file in enumerate(client_files, 1):
        print(f"[{i}] {file.replace('.txt', '')}")
    
    while True:
        try:
            choice = input("\nSkriv siffran för den kund du vill använda: ")
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(client_files):
                selected_file = client_files[choice_idx-1]
                break
        except ValueError:
            pass
        print("Ogiltigt val. Försök igen.")
        
    with open(os.path.join(clients_dir, selected_file), 'r', encoding='utf-8') as f:
        client_context = f.read()
        
    client_name = selected_file.replace('.txt', '').capitalize()
    print(f"\n=> Vald kund: {client_name}\n")
    return client_name, client_context

def generate_gemini_content(user_prompt: str, client_context: str) -> ContentEngineOutput:
    """Step 1: The Brain (Google Gemini API)"""
    logging.info("Step 1: Generating content via Gemini...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    sys_instruction = get_system_instruction(client_context)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            response_mime_type="application/json",
            response_schema=ContentEngineOutput,
            temperature=0.7,
        ),
    )
    
    try:
        parsed_json = json.loads(response.text)
        output = ContentEngineOutput(**parsed_json)
        logging.info("Gemini generation completed.")
        return output
    except Exception as e:
        logging.error(f"Failed to parse Gemini output: {e}")
        logging.error(f"Raw Output: {response.text}")
        raise

async def generate_image_stability(prompt: str, output_path: str) -> str:
    """Step 2a: Generate Image using Stability AI Core"""
    logging.info("Step 2a: Submitting image generation to Stability AI (Core)...")
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "application/json"
    }
    
    data = {"prompt": prompt, "aspect_ratio": "9:16", "output_format": "jpeg"}
    
    try:
        response = await asyncio.to_thread(requests.post, url, headers=headers, files={"none": ''}, data=data)
        
        if response.status_code == 200:
            image_b64 = response.json().get("image")
            # Save local copy for debugging/viewing
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            logging.info("Image generation completed via Stability AI.")
            # Return raw base64 string for Kling
            return image_b64
        else:
            logging.error(f"Stability error: {response.text}")
            raise Exception("Failed to generate image via Stability API")
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        raise

async def generate_audio_elevenlabs(script: str, output_path: str):
    """Step 2b: Generate Voiceover using ElevenLabs"""
    if not USE_ELEVENLABS: return
    
    logging.info("Step 2b: Generating voiceover with ElevenLabs...")
    
    voice_id = "21m00Tcm4TlvDq8ikWAM" 
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    data = {
        "text": script,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    try:
        response = await asyncio.to_thread(requests.post, url, json=data, headers=headers)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            logging.info("Voiceover generation completed.")
        else:
            logging.error(f"ElevenLabs Error: {response.text}")
            raise Exception("Failed to generate voiceover")
    except Exception as e:
        logging.error(f"Audio generation failed: {e}")
        raise

def encode_kling_jwt(ak: str, sk: str) -> str:
    """Generate Kling API JWT token"""
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5
    }
    return jwt.encode(payload, sk, headers=headers)

async def generate_video_kling(image_data: str, prompt: str, output_path: str):
    """Step 3: Generate Video using Kling AI (image2video)"""
    logging.info("Step 3: Submitting video generation to Kling AI...")
    
    token = encode_kling_jwt(KLING_AK, KLING_SK)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 1. Submit task
    submit_url = "https://api.klingai.com/v1/videos/image2video"
    submit_payload = {
        "model": "kling-v1",
        "image": image_data,
        "prompt": prompt,
        "duration": "5"
    }
    
    try:
        submit_res = await asyncio.to_thread(requests.post, submit_url, json=submit_payload, headers=headers)
        submit_data = submit_res.json()
        
        if submit_res.status_code != 200 or submit_data.get('code') != 0:
            logging.error(f"Kling API Submit Error: {submit_data}")
            raise Exception("Failed to submit Kling video task")
            
        task_id = submit_data['data']['task_id']
        logging.info(f"Kling task submitted successfully. Task ID: {task_id}. Polling for completion...")
        
        # 2. Poll for completion
        poll_url = f"https://api.klingai.com/v1/videos/image2video/{task_id}"
        
        while True:
            await asyncio.sleep(10)  # Poll every 10 seconds
            
            # Refresh token to prevent expiration during long renders
            token = encode_kling_jwt(KLING_AK, KLING_SK)
            poll_headers = {"Authorization": f"Bearer {token}"}
            
            poll_res = await asyncio.to_thread(requests.get, poll_url, headers=poll_headers)
            poll_data = poll_res.json()
            
            status = poll_data.get('data', {}).get('task_status')
            if status == 'succeed':
                video_result_raw = poll_data['data']['task_result']['videos'][0]['url']
                logging.info("Kling video generation completed. Downloading...")
                
                # Download the generated silent video
                video_res = await asyncio.to_thread(requests.get, video_result_raw)
                with open(output_path, 'wb') as f:
                    f.write(video_res.content)
                logging.info("Kling video downloaded successfully.")
                break
            elif status == 'failed':
                logging.error(f"Kling video generation failed: {poll_data}")
                raise Exception("Kling video generation failed")
            else:
                logging.info(f"Kling video status: {status}... waiting 10s")
                
    except Exception as e:
        logging.error(f"Kling generation failed: {e}")
        raise

async def upload_to_supabase(client_name: str, content: ContentEngineOutput, video_path: str, image_path: str):
    """Upload video + image to Supabase Storage and log to content_items + content_outputs"""
    if not SUPABASE_SERVICE_KEY:
        logging.warning("SUPABASE_SERVICE_KEY not set. Skipping Supabase upload.")
        return
    
    logging.info("Uploading assets to Supabase...")
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    rest_headers = {**headers, "Content-Type": "application/json", "Prefer": "return=representation"}
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_storage_path = f"{client_name.lower()}/{timestamp}_video.mp4"
    image_storage_path = f"{client_name.lower()}/{timestamp}_thumbnail.jpg"
    
    # Upload video
    try:
        with open(video_path, 'rb') as f:
            upload_res = await asyncio.to_thread(
                requests.post,
                f"{SUPABASE_URL}/storage/v1/object/content-videos/{video_storage_path}",
                headers={**headers, "Content-Type": "video/mp4"},
                data=f
            )
        if upload_res.status_code in [200, 201]:
            logging.info(f"Video uploaded to Supabase Storage: {video_storage_path}")
        else:
            logging.error(f"Video upload failed: {upload_res.text}")
            return
    except Exception as e:
        logging.error(f"Video upload error: {e}")
        return

    # Upload thumbnail
    try:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                await asyncio.to_thread(
                    requests.post,
                    f"{SUPABASE_URL}/storage/v1/object/content-videos/{image_storage_path}",
                    headers={**headers, "Content-Type": "image/jpeg"},
                    data=f
                )
    except Exception as e:
        logging.warning(f"Thumbnail upload error (non-fatal): {e}")

    # Look up client_id from clients table by name
    client_lookup = await asyncio.to_thread(
        requests.get,
        f"{SUPABASE_URL}/rest/v1/clients?name=ilike.{client_name}&select=id&limit=1",
        headers=rest_headers
    )
    client_id = None
    if client_lookup.status_code == 200 and client_lookup.json():
        client_id = client_lookup.json()[0]['id']
        logging.info(f"Matched Supabase client_id: {client_id}")
    else:
        logging.warning(f"No client match found in Supabase for '{client_name}'. Logging without client_id.")

    # Write content_item row
    item_payload = {
        "client_id": client_id,
        "platform": "instagram",
        "content_type": "video",
        "title": f"AI Video - {client_name} - {timestamp}",
        "caption": content.social_caption,
        "script": content.voiceover_script,
        "status": "draft"
    }
    if not client_id:
        item_payload.pop("client_id")

    item_res = await asyncio.to_thread(
        requests.post,
        f"{SUPABASE_URL}/rest/v1/content_items",
        headers=rest_headers,
        json=item_payload
    )
    
    content_item_id = None
    if item_res.status_code in [200, 201] and item_res.json():
        content_item_id = item_res.json()[0]['id']
        logging.info(f"content_items row created: {content_item_id}")

    # Write content_output row
    if content_item_id:
        output_payload = {
            "content_item_id": content_item_id,
            "model_used": "gemini-2.5-flash + stability-core + kling-v1",
            "prompt_used": content.image_prompt,
            "output_type": "video",
            "aspect_ratio": "9:16",
            "approved_status": "draft"
        }
        await asyncio.to_thread(
            requests.post,
            f"{SUPABASE_URL}/rest/v1/content_outputs",
            headers=rest_headers,
            json=output_payload
        )
        logging.info("content_outputs row created.")
    
    # Build public URL for the video
    video_url = f"{SUPABASE_URL}/storage/v1/object/public/content-videos/{video_storage_path}"
    print(f"\n☁️  SUPABASE STORAGE URL: {video_url}")


async def assemble_final_video(video_path: str, audio_path: str, output_path: str):
    """Step 4: Combine video and audio using FFmpeg"""
    if not USE_ELEVENLABS: return
    
    logging.info("Step 4: Assembling final video with FFmpeg...")
    try:
        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        out = ffmpeg.output(video, audio, output_path, vcodec='copy', acodec='aac', strict='experimental')
        await asyncio.to_thread(out.overwrite_output().run, capture_stdout=True, capture_stderr=True)
        logging.info(f"Final video successfully saved to {output_path}")
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg Error: {e.stderr.decode()}")
        raise

async def main():
    print("\n" + "="*50)
    print(" AI CONTENT ENGINE FACTORY")
    print("="*50)
    
    client_name, client_context = choose_client()
    
    user_prompt = input("Vad är idén/budskapet för den här videon? (t.ex. 'Video om 400% tillväxt'):\n> ")
    if not user_prompt.strip():
        logging.error("Tom input. Avbryter.")
        return

    # Ensure /temp directory exists
    temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
    temp_audio_path = os.path.join(temp_dir, "temp_voiceover.mp3")
    temp_video_path = os.path.join(temp_dir, "temp_silent_video.mp4")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_output_path = os.path.join(os.getcwd(), f"{client_name}_Ad_{timestamp}.mp4")
    
    try:
        # ---------------------------------------------------------------------
        # STEP 1: The Brain
        # ---------------------------------------------------------------------
        content = generate_gemini_content(user_prompt, client_context)
        print("\n" + "-"*40)
        print("GEMINI GENERATED CONTENT:")
        print(f"IMAGE PROMPT: {content.image_prompt}")
        print(f"VIDEO PROMPT: {content.video_prompt}")
        if USE_ELEVENLABS:
            print(f"VOICEOVER: {content.voiceover_script}")
        print("-" * 40 + "\n")
        
        # ---------------------------------------------------------------------
        # STEP 2: Asset Generation
        # ---------------------------------------------------------------------
        image_task = asyncio.create_task(generate_image_stability(content.image_prompt, temp_image_path))
        
        if USE_ELEVENLABS and ELEVENLABS_API_KEY:
            audio_task = asyncio.create_task(generate_audio_elevenlabs(content.voiceover_script, temp_audio_path))
        
        # Wait for image to finish (required for video generation)
        image_base64 = await image_task
        
        # ---------------------------------------------------------------------
        # STEP 3: Video Animation (Kling AI)
        # ---------------------------------------------------------------------
        video_task = asyncio.create_task(generate_video_kling(image_base64, content.video_prompt, temp_video_path))
        
        # Wait for tasks to finish
        await video_task
        if USE_ELEVENLABS and ELEVENLABS_API_KEY:
            await audio_task
        
        # ---------------------------------------------------------------------
        # STEP 4: Final Assembly
        # ---------------------------------------------------------------------
        if USE_ELEVENLABS and ELEVENLABS_API_KEY:
            await assemble_final_video(temp_video_path, temp_audio_path, final_output_path)
        else:
            logging.info("Audio disabled. Copying silent video to final output.")
            shutil.copy(temp_video_path, final_output_path)
        
        # Move the image to the main folder instead of deleting it
        final_image_path = final_output_path.replace('.mp4', '.jpg')
        if os.path.exists(temp_image_path):
            shutil.move(temp_image_path, final_image_path)
            logging.info(f"Image thumbnail permanently saved to {final_image_path}")
            
        # Cleanup Temporary Files
        logging.info("Cleaning up temporary files...")
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        
        # ---------------------------------------------------------------------
        # STEP 5: Upload to Supabase
        # ---------------------------------------------------------------------
        await upload_to_supabase(client_name, content, final_output_path, final_image_path)
        
        # Output Results
        print("\n" + "="*50)
        print("🎉 SUCCESS! FINAL AD READY 🎉")
        print(f"File Output: {final_output_path}")
        print("-" * 50)
        print(f"SOCIAL MEDIA CAPTION / HASHTAGS för {client_name}:")
        print(content.social_caption)
        print("=" * 50 + "\n")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    # Validate API Keys before running
    required_keys = [GEMINI_API_KEY, STABILITY_API_KEY, KLING_AK, KLING_SK]
    if USE_ELEVENLABS:
        required_keys.append(ELEVENLABS_API_KEY)
        
    if not all(required_keys):
        logging.error("Missing required API keys in the environment! Please check .env.")
    else:
        asyncio.run(main())
