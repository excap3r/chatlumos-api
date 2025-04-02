import pytest
from unittest.mock import patch, MagicMock
import json

from services.llm_service.llm_service import LLMService
from services.llm_service.providers import LLMProvider, LLMResponse

# Mock provider class
class MockLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str | None = None, name: str = "mock"):
        super().__init__(api_key, model)
        self.name = name
        self._model = model or f"{name}-default-model"
        # Mock methods needed by LLMService
        self.generate = MagicMock()
        self.generate_stream = MagicMock()
        self.get_capabilities = MagicMock(return_value={"mock_capability": True})

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Default mock implementation
        return self.generate(**kwargs)

    def generate_stream(self, prompt: str, **kwargs) -> MagicMock:
         # Default mock implementation
        return self.generate_stream(**kwargs)

    def get_capabilities(self) -> dict:
        return self.get_capabilities()


@pytest.fixture
def mock_config():
    """Provides a basic configuration for LLMService tests."""
    return {
        'DEFAULT_LLM_PROVIDER': 'mock_default',
        'LLM_MODELS': {
            'mock_default': 'mock-model-v1',
            'mock_other': 'other-model-4o'
        },
        'OPENAI_API_KEY': 'fake-openai-key', # Example, mapped internally
        'MOCK_DEFAULT_API_KEY': 'fake-default-key', # For custom mock provider
        'MOCK_OTHER_API_KEY': 'fake-other-key', # For custom mock provider
        'ANTHROPIC_API_KEY': 'fake_anthropic_key',
        'GROQ_API_KEY': 'fake_groq_key',
        'OPENROUTER_API_KEY': 'fake_openrouter_key',
    }

@pytest.fixture
def mock_provider_factory():
    """Mocks the get_llm_provider_factory function."""
    with patch('services.llm_service.llm_service.get_llm_provider_factory') as mock_factory:
        # Default behavior: return a MockLLMProvider instance
        mock_factory.side_effect = lambda provider_name, api_key, model=None: MockLLMProvider(
            api_key=api_key,
            model=model,
            name=provider_name
        )
        yield mock_factory


# --- Initialization Tests ---

def test_llm_service_init_success(mock_config, mock_provider_factory):
    """Test successful initialization with a default provider."""
    service = LLMService(mock_config)
    assert service.default_provider_name == 'mock_default'
    assert service.default_provider is not None
    assert isinstance(service.default_provider, MockLLMProvider)
    assert service.default_provider.name == 'mock_default'
    assert service.default_provider.model == 'mock-model-v1' # From config override
    assert 'mock_default' in service.providers
    assert service.providers['mock_default'] is service.default_provider
    mock_provider_factory.assert_called_once_with(
        'mock_default',
        api_key='fake-default-key', # Assumes internal mapping works or uses direct key name
        model='mock-model-v1'
    )

def test_llm_service_init_factory_error(mock_config, mock_provider_factory):
    """Test initialization failure if the factory raises an error."""
    mock_provider_factory.side_effect = RuntimeError("Factory failed")
    service = LLMService(mock_config)
    assert service.default_provider_name == 'mock_default'
    assert service.default_provider is None # Should be None on failure
    assert 'mock_default' not in service.providers

def test_llm_service_init_missing_default_key(mock_config, mock_provider_factory):
    """Test initialization failure if the default provider's key is missing."""
    config_missing_key = mock_config.copy()
    del config_missing_key['MOCK_DEFAULT_API_KEY'] # Remove the key

    # Adjust mock factory to raise error if key is None (simulating real factory)
    def factory_side_effect(provider_name, api_key, model=None):
        if api_key is None:
             raise ValueError(f"API key not found for {provider_name}")
        return MockLLMProvider(api_key=api_key, model=model, name=provider_name)
    mock_provider_factory.side_effect = factory_side_effect

    service = LLMService(config_missing_key)

    assert service.default_provider is None
    assert 'mock_default' not in service.providers
    # Factory should still be called, but fail internally
    mock_provider_factory.assert_called_once_with(
         'mock_default',
         api_key=None, # The service attempts to get the key, finds None
         model='mock-model-v1'
    )


# --- Provider Management Tests ---

def test_get_provider_instance_default(mock_config, mock_provider_factory):
    """Test getting the already initialized default provider."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock() # Reset after init call

    provider = service._get_provider_instance('mock_default')
    assert provider is service.default_provider
    mock_provider_factory.assert_not_called() # Should use cached instance

def test_get_provider_instance_new(mock_config, mock_provider_factory):
    """Test initializing and caching a new provider instance."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock()

    provider = service._get_provider_instance('mock_other')
    assert provider is not None
    assert isinstance(provider, MockLLMProvider)
    assert provider.name == 'mock_other'
    assert provider.model == 'other-model-4o' # From config override
    assert 'mock_other' in service.providers
    assert service.providers['mock_other'] is provider
    mock_provider_factory.assert_called_once_with(
        'mock_other',
        api_key='fake-other-key', # Assumes internal mapping or direct key name
        model='other-model-4o'
    )

    # Call again, should be cached
    mock_provider_factory.reset_mock()
    provider_cached = service._get_provider_instance('mock_other')
    assert provider_cached is provider
    mock_provider_factory.assert_not_called()

def test_get_provider_instance_unsupported(mock_config, mock_provider_factory):
    """Test getting an unsupported provider raises ValueError."""
    service = LLMService(mock_config)
    with pytest.raises(ValueError, match="Unsupported provider or missing config key mapping: unsupported_provider"):
        service._get_provider_instance('unsupported_provider')

def test_get_provider_instance_missing_key(mock_config, mock_provider_factory):
    """Test getting a provider with missing key raises ValueError."""
    config_missing_key = mock_config.copy()
    # Add a provider name but remove its key
    config_missing_key['LLM_MODELS']['provider_no_key'] = 'some-model'
    # No PROVIDER_NO_KEY_API_KEY defined

    service = LLMService(config_missing_key)
    mock_provider_factory.reset_mock()

    # Adjust factory mock to check for None key
    def factory_side_effect(provider_name, api_key, model=None):
         if api_key is None:
              # This matches the internal logic of _get_provider_instance before calling factory
              raise ValueError(f"API key 'PROVIDER_NO_KEY_API_KEY' not found in configuration for provider 'provider_no_key'.")
         return MockLLMProvider(api_key=api_key, model=model, name=provider_name)
    # We actually expect the service's internal check to raise the error *before* calling the factory
    # when the key is simply missing from the config dict entirely.

    with pytest.raises(ValueError, match="API key 'PROVIDER_NO_KEY_API_KEY' not found"):
         service._get_provider_instance('provider_no_key')

    mock_provider_factory.assert_not_called() # Should fail before factory call


def test_list_providers(mock_config, mock_provider_factory):
    """Test listing initialized providers."""
    service = LLMService(mock_config)
    # Initialize another provider
    service._get_provider_instance('mock_other')
    assert set(service.list_providers()) == {'mock_default', 'mock_other'}

def test_list_providers_init_failure(mock_config, mock_provider_factory):
    """Test listing providers when default init failed."""
    mock_provider_factory.side_effect = RuntimeError("Factory failed")
    service = LLMService(mock_config)
    assert service.list_providers() == []


# --- Capabilities Tests ---

def test_get_capabilities_default(mock_config, mock_provider_factory):
    """Test getting capabilities for the default provider."""
    service = LLMService(mock_config)
    service.default_provider.get_capabilities.return_value = {"default_caps": True} # type: ignore

    caps = service.get_capabilities('mock_default')
    assert caps == {"default_caps": True}
    service.default_provider.get_capabilities.assert_called_once() # type: ignore

def test_get_capabilities_specific(mock_config, mock_provider_factory):
    """Test getting capabilities for a specific, newly initialized provider."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock() # Reset after init

    # Configure the mock factory to return a provider mock we can control
    other_provider_mock = MockLLMProvider(api_key="other", name="mock_other")
    other_provider_mock.get_capabilities.return_value = {"other_caps": 123}
    mock_provider_factory.side_effect = lambda *args, **kwargs: other_provider_mock if kwargs.get('name') == 'mock_other' else MockLLMProvider(*args, **kwargs)
    # Re-patching side effect might be tricky, let's adjust the instance after getting it
    provider_instance = service._get_provider_instance('mock_other') # This uses the factory
    provider_instance.get_capabilities.return_value = {"other_caps": 123} # type: ignore

    caps = service.get_capabilities('mock_other')
    assert caps == {"other_caps": 123}
    provider_instance.get_capabilities.assert_called_once() # type: ignore

def test_get_capabilities_specific_error(mock_config, mock_provider_factory):
    """Test getting capabilities when the provider instance fails."""
    service = LLMService(mock_config)
    provider_instance = service._get_provider_instance('mock_other')
    error_message = "Provider capability check failed"
    provider_instance.get_capabilities.side_effect = RuntimeError(error_message) # type: ignore

    caps = service.get_capabilities('mock_other')
    assert caps == {"error": error_message}

def test_get_capabilities_init_error(mock_config, mock_provider_factory):
    """Test getting capabilities for a provider that failed initialization."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock()
    error_message = "Cannot initialize this provider"
    mock_provider_factory.side_effect = ValueError(error_message)

    caps = service.get_capabilities('failed_provider') # Try to get a non-initialized provider
    assert "error" in caps
    # Check that the error message reflects the *initialization* failure
    assert "API key 'FAILED_PROVIDER_API_KEY' not found" in caps["error"] or error_message in caps["error"]


def test_get_capabilities_all(mock_config, mock_provider_factory):
    """Test getting capabilities for all initialized providers."""
    service = LLMService(mock_config)
    service.default_provider.get_capabilities.return_value = {"default": True} # type: ignore

    # Initialize another provider and set its capabilities
    other_provider = service._get_provider_instance('mock_other')
    other_provider.get_capabilities.return_value = {"other": 1} # type: ignore

    all_caps = service.get_capabilities()
    assert all_caps == {
        'mock_default': {"default": True},
        'mock_other': {"other": 1}
    }
    service.default_provider.get_capabilities.assert_called_once() # type: ignore
    other_provider.get_capabilities.assert_called_once() # type: ignore

def test_get_capabilities_all_with_error(mock_config, mock_provider_factory):
    """Test getting all capabilities when one provider fails."""
    service = LLMService(mock_config)
    service.default_provider.get_capabilities.return_value = {"default": True} # type: ignore

    other_provider = service._get_provider_instance('mock_other')
    error_msg = "Capability check failed"
    other_provider.get_capabilities.side_effect = RuntimeError(error_msg) # type: ignore

    all_caps = service.get_capabilities()
    assert all_caps == {
        'mock_default': {"default": True},
        'mock_other': {"error": f"Failed to get capabilities: {error_msg}"}
    }


# --- decompose_question Tests ---

def test_decompose_question_success_default_provider(mock_config, mock_provider_factory):
    """Test successful decomposition using the default provider."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    question = "What are the main causes and effects of climate change?"
    expected_subquestions = ["What are the primary greenhouse gases?", "What are the impacts on sea level?"]

    # Mock the provider's generate method
    mock_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=json.dumps(expected_subquestions),
        is_error=False,
        parsed_content=None # Service should populate this
    )
    mock_default_provider.generate.return_value = mock_response # type: ignore

    result = service.decompose_question(question)

    assert not result.is_error
    assert result.provider == 'mock_default'
    assert result.model == 'mock-model-v1'
    assert result.content == json.dumps(expected_subquestions)
    assert result.parsed_content == expected_subquestions
    mock_default_provider.generate.assert_called_once() # type: ignore
    call_args, call_kwargs = mock_default_provider.generate.call_args # type: ignore
    assert call_kwargs['prompt'].startswith(f"Original question: {question}")
    assert "Return your response STRICTLY as a JSON array" in call_kwargs['system_prompt']
    assert call_kwargs['temperature'] == 0.2
    assert call_kwargs['model'] is None # No specific model override requested

def test_decompose_question_success_specific_provider_and_model(mock_config, mock_provider_factory):
    """Test successful decomposition using a specific provider and model."""
    service = LLMService(mock_config)
    mock_other_provider = service._get_provider_instance('mock_other') # Initialize it
    question = "Compare Python and Javascript for web development."
    expected_subquestions = ["Frontend capabilities?", "Backend frameworks?", "Performance?"]
    specific_model = "other-model-4o-mini" # Request a specific sub-model

    mock_response = LLMResponse(
        provider=mock_other_provider.name,
        model=specific_model, # Provider should echo the requested model
        content=json.dumps(expected_subquestions),
        is_error=False
    )
    mock_other_provider.generate.return_value = mock_response # type: ignore
    mock_provider_factory.reset_mock() # Reset factory mock after init

    result = service.decompose_question(question, provider_name='mock_other', model=specific_model)

    assert not result.is_error
    assert result.provider == 'mock_other'
    assert result.model == specific_model
    assert result.parsed_content == expected_subquestions
    mock_other_provider.generate.assert_called_once() # type: ignore
    call_args, call_kwargs = mock_other_provider.generate.call_args # type: ignore
    assert call_kwargs['prompt'].startswith(f"Original question: {question}")
    assert call_kwargs['model'] == specific_model # Check model override was passed
    # Factory should not be called again as provider was cached
    mock_provider_factory.assert_not_called()

def test_decompose_question_provider_init_error(mock_config, mock_provider_factory):
    """Test decompose_question when the requested provider fails to initialize."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock()
    error_message = "Cannot initialize mock_other"
    mock_provider_factory.side_effect = ValueError(error_message)

    result = service.decompose_question("Any question", provider_name='mock_other')

    assert result.is_error
    assert result.error == error_message
    assert result.provider == 'mock_other'
    assert result.model is None # Model wasn't specified
    assert result.parsed_content is None

def test_decompose_question_provider_generate_error(mock_config, mock_provider_factory):
    """Test decompose_question when the provider's generate method fails."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    error_message = "LLM API error"

    # Mock the provider's generate method to return an error response
    error_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=None,
        is_error=True,
        error=error_message
    )
    mock_default_provider.generate.return_value = error_response # type: ignore

    result = service.decompose_question("Any question")

    assert result.is_error
    assert result.error == error_message
    assert result.provider == 'mock_default'
    assert result.model == 'mock-model-v1'
    assert result.parsed_content is None

def test_decompose_question_parse_json_in_markdown(mock_config, mock_provider_factory):
    """Test parsing JSON successfully when it's embedded in markdown code blocks."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    question = "Explain black holes."
    expected_subquestions = ["What is singularity?", "What is event horizon?"]
    response_content = f"""
Here are the sub-questions:
```json
{json.dumps(expected_subquestions)}
```
Let me know if you need more.
"""

    mock_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=response_content,
        is_error=False
    )
    mock_default_provider.generate.return_value = mock_response # type: ignore

    result = service.decompose_question(question)

    assert not result.is_error
    assert result.content == response_content # Original content is preserved
    assert result.parsed_content == expected_subquestions # Parsed content is extracted

def test_decompose_question_non_json_response(mock_config, mock_provider_factory):
    """Test handling when the LLM response is not JSON and cannot be parsed."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    response_content = "Sorry, I cannot break that down. It seems simple enough."

    mock_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=response_content,
        is_error=False # The LLM call itself didn't fail
    )
    mock_default_provider.generate.return_value = mock_response # type: ignore

    result = service.decompose_question("Is the sky blue?")

    assert result.is_error
    assert result.error == "LLM response was not in the expected JSON format."
    assert result.content == response_content
    assert result.parsed_content is None

def test_decompose_question_json_not_a_list(mock_config, mock_provider_factory):
    """Test handling when the LLM returns valid JSON, but it's not a list."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    response_content = json.dumps({"question": "q1?", "details": "d1"})

    mock_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=response_content,
        is_error=False
    )
    mock_default_provider.generate.return_value = mock_response # type: ignore

    result = service.decompose_question("A question")

    assert result.is_error
    assert result.error == "LLM response was valid JSON but not a list as expected."
    assert result.content == response_content
    assert result.parsed_content is None

def test_decompose_question_invalid_json_in_markdown(mock_config, mock_provider_factory):
    """Test handling when JSON in markdown is detected but is invalid."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    response_content = '```json\n[ "Question 1?", "Question 2" \' Syntax Error Here \n```'

    mock_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=response_content,
        is_error=False
    )
    mock_default_provider.generate.return_value = mock_response # type: ignore

    result = service.decompose_question("A question")

    assert result.is_error
    assert "Failed to parse JSON extracted from markdown" in result.error
    assert result.content == response_content
    assert result.parsed_content is None 


# --- generate_answer Tests (Non-Streaming) ---

def test_generate_answer_success(mock_config, mock_provider_factory):
    """Test successful answer generation (non-streaming)."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    question = "What is the capital of France?"
    search_results = [
        {"id": "doc1", "content": "Paris is the capital and largest city of France.", "score": 0.9},
        {"id": "doc2", "content": "France is a country in Europe.", "score": 0.5}
    ]
    expected_answer = "The capital of France is Paris."

    # Mock the provider's generate method
    mock_response = LLMResponse(
        provider=mock_default_provider.name, # type: ignore
        model=mock_default_provider.model, # type: ignore
        content=expected_answer,
        is_error=False
    )
    mock_default_provider.generate.return_value = mock_response # type: ignore

    result = service.generate_answer(question, search_results)

    assert isinstance(result, LLMResponse)
    assert not result.is_error
    assert result.provider == 'mock_default'
    assert result.model == 'mock-model-v1'
    assert result.content == expected_answer
    mock_default_provider.generate.assert_called_once() # type: ignore
    call_args, call_kwargs = mock_default_provider.generate.call_args # type: ignore
    assert call_kwargs['question'] == question
    assert call_kwargs['search_results'] == search_results
    assert call_kwargs['model'] is None # No specific model requested

def test_generate_answer_provider_init_error(mock_config, mock_provider_factory):
    """Test generate_answer failure when the provider cannot be initialized."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock()
    error_message = "Cannot initialize this provider"
    mock_provider_factory.side_effect = ValueError(error_message)

    result = service.generate_answer("Q", [], provider_name='uninit_provider')

    assert isinstance(result, LLMResponse)
    assert result.is_error
    assert result.error == error_message
    assert result.provider == 'uninit_provider'

def test_generate_answer_provider_generate_error(mock_config, mock_provider_factory):
    """Test generate_answer failure when the provider's generate method returns an error."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    error_message = "LLM generation failed"

    error_response = LLMResponse(
        provider=mock_default_provider.name, model=mock_default_provider.model, is_error=True, error=error_message # type: ignore
    )
    mock_default_provider.generate.return_value = error_response # type: ignore

    result = service.generate_answer("Q", [])

    assert isinstance(result, LLMResponse)
    assert result.is_error
    assert result.error == error_message
    assert result.provider == 'mock_default'


# --- generate_answer Tests (Streaming) ---

def test_generate_answer_stream_success(mock_config, mock_provider_factory):
    """Test successful streaming answer generation."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    question = "Explain photosynthesis."
    search_results = [{"id": "bio1", "content": "It's how plants make food.", "score": 0.8}]
    stream_chunks = [
        LLMResponse(provider="mock_default", model="mock-model-v1", chunk="Photo", is_chunk=True),
        LLMResponse(provider="mock_default", model="mock-model-v1", chunk="synthesis is", is_chunk=True),
        LLMResponse(provider="mock_default", model="mock-model-v1", chunk=" the process...", is_chunk=True, is_final_chunk=True, content="Photosynthesis is the process...") # Final chunk includes full content
    ]

    # Mock the provider's generate_stream method to return a generator
    def mock_stream_generator(**kwargs):
        yield from stream_chunks
    mock_default_provider.generate_stream.side_effect = mock_stream_generator # type: ignore

    # Use a mock StreamingHandler
    mock_handler = MagicMock(spec=StreamingHandler)

    result_generator = service.generate_answer(
        question, search_results, stream=True, stream_handler=mock_handler
    )

    # Consume the generator and check results
    results_list = list(result_generator)

    assert len(results_list) == len(stream_chunks)
    for i, chunk in enumerate(results_list):
        assert chunk is stream_chunks[i] # Check identity

    # Verify the provider's stream method was called correctly
    mock_default_provider.generate_stream.assert_called_once() # type: ignore
    call_args, call_kwargs = mock_default_provider.generate_stream.call_args # type: ignore
    assert call_kwargs['question'] == question
    assert call_kwargs['search_results'] == search_results
    assert call_kwargs['stream_handler'] is mock_handler # Check handler was passed
    assert call_kwargs['model'] is None

def test_generate_answer_stream_provider_init_error(mock_config, mock_provider_factory):
    """Test streaming failure when the provider cannot be initialized."""
    service = LLMService(mock_config)
    mock_provider_factory.reset_mock()
    error_message = "Cannot initialize stream provider"
    mock_provider_factory.side_effect = ValueError(error_message)

    result_generator = service.generate_answer("Q", [], provider_name='uninit_stream', stream=True)

    # Consume the generator
    results_list = list(result_generator)

    assert len(results_list) == 1
    result = results_list[0]
    assert isinstance(result, LLMResponse)
    assert result.is_error
    assert result.error == error_message
    assert result.provider == 'uninit_stream'

def test_generate_answer_stream_provider_generate_error(mock_config, mock_provider_factory):
    """Test streaming failure when the provider's stream method yields an error."""
    service = LLMService(mock_config)
    mock_default_provider = service.default_provider
    error_message = "LLM streaming failed mid-way"

    error_response = LLMResponse(
        provider=mock_default_provider.name, model=mock_default_provider.model, is_error=True, error=error_message # type: ignore
    )

    # Mock the provider's generate_stream to yield the error
    def mock_stream_generator(**kwargs):
        yield LLMResponse(provider="mock_default", model="mock-model-v1", chunk="First part.", is_chunk=True)
        yield error_response
    mock_default_provider.generate_stream.side_effect = mock_stream_generator # type: ignore

    result_generator = service.generate_answer("Q", [], stream=True)

    # Consume the generator
    results_list = list(result_generator)

    assert len(results_list) == 2
    assert not results_list[0].is_error
    assert results_list[0].chunk == "First part."
    assert results_list[1].is_error
    assert results_list[1].error == error_message
    assert results_list[1].provider == 'mock_default' 