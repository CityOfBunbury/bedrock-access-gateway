import os
import json
import time
import logging
import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ..schema import ChatRequest, Models, Model
from ..schema import (
    ChatResponse,
    ChatResponseMessage,
    ChatStreamResponse,
    Choice,
    ChoiceDelta,
    Usage,
    ErrorMessage,
    Error
)
from ..setting import AGENTS, DEFAULT_AGENT, AWS_REGION

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize Bedrock client (consider moving to a shared client module later)
# For now, initializing here. Ensure AWS credentials are configured.
try:
    bedrock_agent_runtime = boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=AWS_REGION,
        # Assuming credentials are handled by environment, IAM role, or shared config
    )
except Exception as e:
    logger.error(f"Error initializing Bedrock Agent Runtime client: {e}")
    bedrock_agent_runtime = None # Handle appropriately if client fails to initialize

@router.post("/chat/completions", summary="Proxy OpenAI-style chat completions to Bedrock Agent")
async def agent_chat_completions(request: Request):
    """
    Handles OpenAI-style chat completions requests and proxies them to the configured Bedrock Agent.
    """
    if not bedrock_agent_runtime:
        raise HTTPException(status_code=500, detail="Bedrock Agent Runtime client not initialized.")
        
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
        logger.debug(f"Received agent chat request: {body}")

        model_id = chat_request.model or DEFAULT_AGENT
        stream_mode = chat_request.stream

        # Get agent configuration
        agent_config = AGENTS.get(model_id)
        if not agent_config:
            if not DEFAULT_AGENT:
                 raise HTTPException(status_code=400, detail=f"No default agent configured and agent '{model_id}' not found.")
            logger.warning(f"Unknown agent model ID: {model_id}, using default agent {DEFAULT_AGENT}")
            agent_config = AGENTS.get(DEFAULT_AGENT)
            model_id = DEFAULT_AGENT # Use the default agent's name as model_id in response

        if not agent_config: # If default agent also not found after check
             raise HTTPException(status_code=500, detail="Default agent configuration not found.")

        agent_id = agent_config["agent_id"]
        alias_id = agent_config["alias_id"]

        # Construct input from all messages
        full_input = ""
        if chat_request.messages:
            # Simple concatenation
            full_input = "\\n\\n".join([f"{msg.role}: {msg.content}" for msg in chat_request.messages if msg.content]) # Join non-empty content with roles

        if not full_input: # Check if after concatenation, input is still empty
             raise HTTPException(status_code=400, detail="No message content found in the request.")

        # Use a random session ID - we will handle conversation history outside of bedrock
        session_id = body.get("session_id", f"session-{model_id}-{os.urandom(4).hex()}")

        agent_request = {
            "agentId": agent_id,
            "agentAliasId": alias_id,
            "sessionId": session_id,
            "inputText": full_input,
            "enableTrace": False # Set to True for debugging if needed
        }

        logger.info(f"Invoking Bedrock Agent: {agent_id} (Alias: {alias_id}) with session: {session_id}")

        # Call Bedrock Agent
        try:
            response = bedrock_agent_runtime.invoke_agent(**agent_request)
        except ClientError as e:
            error_message = f"AWS Bedrock Agent error: {e.response['Error']['Message']}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)
        except Exception as e:
            logger.error(f"Error invoking Bedrock Agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        response_id = f"chatcmpl-agent-{model_id}-{os.urandom(4).hex()}"
        created_time = int(time.time())

        # Handle Streaming Response
        if stream_mode:
            async def stream_generator():
                try:
                    # Initial chunk with role
                    initial_chunk = ChatStreamResponse(
                        id=response_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=model_id,
                        choices=[ChoiceDelta(index=0, delta={"role": "assistant"}, finish_reason=None)],
                        usage=None  # Usage is typically null in non-final chunks
                    )
                    yield f"data: {initial_chunk.model_dump_json()}\n\n"

                    for event in response['completion']:
                        if 'chunk' in event:
                            content = event['chunk']['bytes'].decode('utf-8')
                            if content:
                                chunk_obj = ChatStreamResponse(
                                    id=response_id,
                                    object="chat.completion.chunk",
                                    created=created_time,
                                    model=model_id,
                                    choices=[ChoiceDelta(index=0, delta={"content": content}, finish_reason=None)],
                                    usage=None
                                )
                                yield f"data: {chunk_obj.model_dump_json()}\n\n"
                        elif 'trace' in event:
                             # log trace information
                             logger.debug(f"Agent trace: {event['trace']}")

                    # Final chunk with finish reason
                    final_chunk = ChatStreamResponse(
                        id=response_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=model_id,
                        choices=[ChoiceDelta(index=0, delta={}, finish_reason="stop")],
                         # TODO: Add usage here if stream_options.include_usage is implemented
                        usage=None
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error during agent stream generation: {e}")
                    # Send an error chunk to the client (using the Error schema might be better if needed)
                    error_detail = ErrorMessage(message=f'Stream generation error: {e}', type='stream_error')
                    error_chunk = Error(error=error_detail)
                    # Note: OpenAI stream errors aren't standard, JSON format might vary.
                    # This sends a JSON object, adjust if a specific error format is needed.
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"


            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # Handle Non-Streaming Response
        else:
            completion = ""
            try:
                for event in response['completion']:
                    if 'chunk' in event:
                        completion += event['chunk']['bytes'].decode('utf-8')
                    elif 'trace' in event:
                        logger.debug(f"Agent trace: {event['trace']}") # Log trace info
            except Exception as e:
                 logger.error(f"Error processing agent response completion: {e}")
                 raise HTTPException(status_code=500, detail="Error processing agent response.")

            # Use Pydantic models for the response
            openai_response = ChatResponse(
                id=response_id,
                object="chat.completion",
                created=created_time,
                model=model_id,
                choices=[
                    Choice(
                        index=0,
                        message=ChatResponseMessage(
                            role="assistant",
                            content=completion
                        ),
                        finish_reason="stop" # Bedrock Agent likely stops when done
                    )
                ],
                # Usage data not directly available from Bedrock Agent invoke_agent
                # Use the Usage model with placeholder values
                usage=Usage(
                    prompt_tokens=-1,
                    completion_tokens=-1,
                    total_tokens=-1
                )
            )
            logger.info(f"Sending non-streaming response from agent {model_id}")
            # FastAPI automatically handles Pydantic model serialization
            return openai_response

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in agent chat completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/models", response_model=Models, summary="List available Bedrock Agents as models")
async def list_agent_models():
    """
    Returns a list of configured Bedrock Agents, formatted like OpenAI models.
    """
    model_cards = []
    for model_id, _ in AGENTS.items():
        model_cards.append(Model(
            id=model_id,
            object="model",
            owned_by="amazon-bedrock-agent", # Indicate it's an agent
            permission=[], # Permissions not applicable in this context
            created=int(time.time()), # Use current time
        ))
        
    if not model_cards and DEFAULT_AGENT and DEFAULT_AGENT in AGENTS: # Ensure default is listed if it exists but loop was empty
        # This case shouldn't happen if AGENTS is populated correctly, but as a safeguard
         model_cards.append(Model(
            id=DEFAULT_AGENT,
            object="model",
            owned_by="amazon-bedrock-agent",
            permission=[], 
            created=int(time.time()),
        ))

    logger.info(f"Listing available agents: {list(AGENTS.keys())}")
    return Models(data=model_cards) 