"""Tests for named custom provider and 'main' alias resolution in auxiliary_client."""

import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Write a minimal config so load_config doesn't fail
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


class TestNormalizeVisionProvider:
    """_normalize_vision_provider should resolve 'main' to actual main provider."""


    def test_main_resolves_to_openrouter(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "openrouter"






    def test_auto_unchanged(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("auto") == "auto"
        assert _normalize_vision_provider(None) == "auto"


class TestResolveProviderClientMainAlias:
    """resolve_provider_client('main', ...) should resolve to actual main provider."""

    def test_main_resolves_to_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "override-model")
        assert client is not None
        assert model == "override-model"
        assert "beans.local" in str(client.base_url)

    def test_main_with_custom_colon_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "custom:beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "test")
        assert client is not None
        assert "beans.local" in str(client.base_url)

    def test_main_resolves_github_copilot_alias(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "gpt-5.4", "provider": "github-copilot"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "ghu_test_token",
                "base_url": "https://api.githubcopilot.com",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_openai.called


class TestResolveProviderClientNamedCustom:
    """resolve_provider_client should resolve named custom providers directly."""

    def test_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("beans", "my-model")
        assert client is not None
        assert model == "my-model"
        assert "beans.local" in str(client.base_url)


    def test_named_custom_no_api_key_uses_fallback(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test"},
            "custom_providers": [
                {"name": "local", "base_url": "http://localhost:8080/v1"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("local", "test")
        assert client is not None
        # no-key-required should be used

    def test_providers_dict_uses_durable_pool_when_no_inline_key(self, tmp_path):
        """Titles/compression/vision must read credential_pool.<key>, not a placeholder."""
        _write_config(tmp_path, {
            "providers": {
                "b-ai": {
                    "name": "B.AI",
                    "base_url": "https://api.b.ai/v1",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "b-ai": [
                    {
                        "id": "k1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-real-b-ai-pool-key-12345",
                    }
                ]
            },
        }))
        from agent.auxiliary_client import resolve_provider_client
        client, _model = resolve_provider_client("b-ai", "b-ai-model")
        assert client is not None
        assert "api.b.ai" in str(client.base_url)
        assert client.api_key == "sk-real-b-ai-pool-key-12345"


class TestResolveProviderClientModelNormalization:
    """Direct-provider auxiliary routing should normalize models like main runtime."""

    def test_matching_native_prefix_is_stripped_for_main_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "zai/glm-5.1")

        assert client is not None
        assert model == "glm-5.1"


    def test_aggregator_vendor_slug_is_preserved(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "openrouter", "anthropic/claude-sonnet-4.6"
            )

        assert client is not None
        assert model == "anthropic/claude-sonnet-4.6"


class TestResolveVisionProviderClientModelNormalization:
    """Vision auto-routing should reuse the same provider-specific normalization."""

    def test_vision_auto_strips_matching_main_provider_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "zai"
        assert client is not None
        assert model == "glm-5v-turbo"  # zai has dedicated vision model in _PROVIDER_VISION_MODELS


class TestVisionPathApiMode:
    """Vision path should propagate api_mode to _get_cached_client."""

    def test_explicit_provider_passes_api_mode(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "auxiliary": {"vision": {"api_mode": "chat_completions"}},
        })
        with patch("agent.auxiliary_client._get_cached_client") as mock_gcc:
            mock_gcc.return_value = (MagicMock(), "test-model")
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client(provider="deepseek")

        mock_gcc.assert_called_once()
        _, kwargs = mock_gcc.call_args
        assert kwargs.get("api_mode") == "chat_completions"


class TestProvidersDictApiModeAnthropicMessages:
    """Regression guard for #15033.

    Named providers declared under the ``providers:`` dict with
    ``api_mode: anthropic_messages`` must route auxiliary calls through
    the Anthropic Messages API (via AnthropicAuxiliaryClient), not
    through an OpenAI chat-completions client.

    The bug had two halves: the providers-dict branch of
    ``_get_named_custom_provider`` dropped the ``api_mode`` field, and
    ``resolve_provider_client``'s named-custom branch never read it.
    """

    def test_providers_dict_propagates_api_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                },
            },
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("myrelay")
        assert entry is not None
        assert entry.get("api_mode") == "anthropic_messages"
        assert entry.get("base_url") == "https://example-relay.test/anthropic"
        assert entry.get("api_key") == "sk-test"



    def test_resolve_provider_client_returns_anthropic_client(self, tmp_path, monkeypatch):
        """Named custom provider with api_mode=anthropic_messages must
        route through AnthropicAuxiliaryClient."""
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                },
            },
        })
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
            AsyncAnthropicAuxiliaryClient,
        )
        sync_client, sync_model = resolve_provider_client("myrelay", async_mode=False)
        assert isinstance(sync_client, AnthropicAuxiliaryClient), (
            f"expected AnthropicAuxiliaryClient, got {type(sync_client).__name__}"
        )
        assert sync_model == "claude-opus-4-7"

        async_client, async_model = resolve_provider_client("myrelay", async_mode=True)
        assert isinstance(async_client, AsyncAnthropicAuxiliaryClient), (
            f"expected AsyncAnthropicAuxiliaryClient, got {type(async_client).__name__}"
        )
        assert async_model == "claude-opus-4-7"




class TestCustomProviderAliasCollision:
    """A user-declared custom_providers entry whose name matches a built-in
    *alias* (not a canonical provider) must win over the built-in.

    Regression guard for #15743: users who defined fallback_model pointing at
    a custom_providers entry named ``kimi`` were having requests routed to
    the built-in kimi-coding endpoint because ``_normalize_aux_provider``
    rewrote ``kimi`` → ``kimi-coding`` before the named-custom lookup.
    """

    def test_custom_named_kimi_wins_over_builtin_alias(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
            "custom_providers": [
                {
                    "name": "kimi",
                    "base_url": "https://my-custom-kimi.example.com/v1",
                    "api_key": "my-kimi-key",
                    "models": {"my-kimi-model": {"context_length": 200000}},
                },
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI
        client, model = resolve_provider_client("kimi", model="my-kimi-model", raw_codex=True)
        assert isinstance(client, OpenAI)
        assert "my-custom-kimi.example.com" in str(client.base_url)
        assert client.api_key == "my-kimi-key"
        assert model == "my-kimi-model"

    def test_bare_kimi_without_custom_still_routes_to_builtin(self, tmp_path, monkeypatch):
        """Regression guard: bare 'kimi' with no custom entry must still
        reach the built-in kimi-coding provider."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        monkeypatch.setenv("KIMI_API_KEY", "builtin-kimi-key")
        from agent.auxiliary_client import resolve_provider_client
        client, _ = resolve_provider_client("kimi", model="kimi-k2-0905-preview", raw_codex=True)
        assert client is not None
        base_url = str(client.base_url)
        # Built-in kimi-coding points at api.moonshot.ai
        assert "moonshot" in base_url or "kimi" in base_url, f"unexpected base_url {base_url!r}"

    def test_explicit_overrides_applied_on_api_key_branch(self, tmp_path, monkeypatch):
        """Explicit base_url/api_key from the caller must override the
        registered provider's defaults on the API-key branch.  Used by
        _try_activate_fallback to route a fallback through a built-in
        provider name but targeting a user-supplied endpoint."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        monkeypatch.setenv("KIMI_API_KEY", "builtin-kimi-key")
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI
        client, _ = resolve_provider_client(
            "kimi-coding", model="kimi-k2", raw_codex=True,
            explicit_base_url="https://override.example.com",
            explicit_api_key="override-key",
        )
        assert isinstance(client, OpenAI)
        assert "override.example.com" in str(client.base_url)
        assert client.api_key == "override-key"


class TestResolveProviderClientMainRuntimeCustom:
    """When the main agent uses a named custom provider (custom:<name>),
    resolve_provider_client('custom', ..., main_runtime=...) must reuse the
    main_runtime's base_url + api_key instead of re-resolving from the bare
    'custom' provider name.  Re-resolution loses the provider name and falls
    back to OpenRouter or a wrong API-key provider. (#45472)"""

    def test_custom_provider_main_runtime_used_directly(self, tmp_path, monkeypatch):
        """main_runtime with base_url + api_key for a named custom provider
        is used directly, bypassing the _try_custom_endpoint / API-key
        fallback chain."""
        from agent.auxiliary_client import resolve_provider_client
        main_runtime = {
            "provider": "custom",
            "base_url": "https://my-gateway.example.com/v1",
            "api_key": "***",
            "model": "glm-5.1",
        }
        client, model = resolve_provider_client(
            "custom",
            model="explicit-glm-5.1",
            main_runtime=main_runtime,
        )
        assert client is not None
        assert model == "explicit-glm-5.1"
        assert "my-gateway.example.com" in str(client.base_url)
        assert client.api_key == "***"

    def test_custom_provider_main_runtime_preserves_responses_mode_for_compression(self, monkeypatch):
        """An automatic compression override of ``provider: custom`` must retain
        the live main runtime's wire mode.  The named Tianji provider resolves
        to a bare ``custom`` runtime, so dropping ``api_mode`` here silently
        sends the summary to ``/chat/completions`` instead of ``/responses``.
        """
        from agent.auxiliary_client import CodexAuxiliaryClient, resolve_provider_client

        real_client = MagicMock()
        real_client.api_key = "***"
        real_client.base_url = "https://tianji.example.test/v1"
        monkeypatch.setattr(
            "agent.auxiliary_client._create_openai_client",
            lambda **_kwargs: real_client,
        )

        client, model = resolve_provider_client(
            "custom",
            model="gpt-5.6-luna",
            task="compression",
            main_runtime={
                "provider": "custom",
                "model": "gpt-5.6-luna",
                "base_url": "https://tianji.example.test/v1",
                "api_key": "***",
                "api_mode": "codex_responses",
            },
        )

        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.6-luna"
        assert client._real_client is real_client

    def test_custom_provider_main_runtime_mode_separates_cached_transport(self, monkeypatch):
        """Changing the inherited main-runtime wire mode must not reuse a cached adapter."""
        from agent.auxiliary_client import CodexAuxiliaryClient, _client_cache, _get_cached_client

        created_clients = []

        def make_client(**kwargs):
            client = MagicMock()
            client.base_url = kwargs.get("base_url")
            created_clients.append(client)
            return client

        monkeypatch.setattr(
            "agent.auxiliary_client._create_openai_client",
            make_client,
        )
        _client_cache.clear()
        runtime = {
            "provider": "custom",
            "model": "gpt-5.6-luna",
            "base_url": "https://tianji.example.test/v1",
            "api_key": "***",
        }

        responses_client, _ = _get_cached_client(
            "custom", model="gpt-5.6-luna", task="compression",
            main_runtime={**runtime, "api_mode": "codex_responses"},
        )
        chat_client, _ = _get_cached_client(
            "custom", model="gpt-5.6-luna", task="compression",
            main_runtime={**runtime, "api_mode": "chat_completions"},
        )
        other_client, _ = _get_cached_client(
            "custom", model="gpt-5.6-luna", task="compression",
            main_runtime={
                **runtime,
                "base_url": "https://other.tianji.example.test/v1",
                "api_key": "***-other",
                "api_mode": "chat_completions",
            },
        )

        assert isinstance(responses_client, CodexAuxiliaryClient)
        assert chat_client is created_clients[1]
        assert chat_client is not responses_client
        assert other_client is created_clients[2]
        assert other_client is not chat_client

    def test_custom_provider_inherited_mode_reaches_relay_metadata(self, monkeypatch):
        """The inherited mode must reach Relay's protocol/codec selection."""
        import agent.auxiliary_client as auxiliary_client

        runtime = {
            "provider": "custom",
            "model": "gpt-5.6-luna",
            "base_url": "https://tianji.example.test/v1",
            "api_key": "***",
            "api_mode": "codex_responses",
        }
        client = MagicMock()
        client.base_url = runtime["base_url"]
        monkeypatch.setattr(
            auxiliary_client,
            "_resolve_task_provider_model",
            lambda *args, **kwargs: ("custom", "gpt-5.6-luna", None, None, None),
        )
        monkeypatch.setattr(
            auxiliary_client,
            "_get_cached_client",
            lambda *args, **kwargs: (client, "gpt-5.6-luna"),
        )
        with patch.object(auxiliary_client, "_set_relay_auxiliary_route") as set_route:
            auxiliary_client._prepare_aux_request(
                "title_generation", provider=None, model=None, base_url=None, api_key=None,
                main_runtime=runtime, messages=[{"role": "user", "content": "x"}],
                temperature=None, max_tokens=None, tools=None, timeout=None, extra_body=None,
                reasoning_config=None, extra_headers=None, api_mode=None, route_info={},
                async_mode=False,
            )

        assert set_route.call_args.args[2] == "codex_responses"


    def test_custom_provider_main_runtime_no_credentials_falls_through(self, tmp_path, monkeypatch):
        """When main_runtime has no base_url or no api_key, the existing
        _try_custom_endpoint / _resolve_api_key_provider fallback chain is
        still tried."""
        # Ensure no env-provided credentials interfere
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from agent.auxiliary_client import resolve_provider_client
        # main_runtime with key but no base_url → must fall through
        client, model = resolve_provider_client(
            "custom",
            main_runtime={"api_key": "k", "base_url": ""},
        )
        # Should fall through to _try_custom_endpoint → return None,None
        # because no OPENAI_BASE_URL is set and no custom endpoint is configured
        assert client is None

    def test_custom_provider_main_runtime_respects_explicit_base_url(self, tmp_path):
        """explicit_base_url still wins over main_runtime — the caller's
        explicit argument is the strongest signal."""
        from agent.auxiliary_client import resolve_provider_client, CodexAuxiliaryClient
        main_runtime = {
            "base_url": "https://main-runtime.example.com/v1",
            "api_key": "sk-main",
            "model": "ignored-model",
            "api_mode": "codex_responses",
        }
        client, model = resolve_provider_client(
            "custom",
            model="explicit-model",
            explicit_base_url="https://explicit.example.com/v1",
            explicit_api_key="sk-explicit",
            main_runtime=main_runtime,
        )
        assert client is not None
        assert model == "explicit-model"
        assert "explicit.example.com" in str(client.base_url)
        assert client.api_key == "sk-explicit"
        assert not isinstance(client, CodexAuxiliaryClient)

    def test_automatic_compression_inherited_custom_responses_dispatches_stream(self, tmp_path, monkeypatch):
        """Automatic compression must reach the inherited custom Responses wire."""
        from types import SimpleNamespace

        import agent.auxiliary_client as auxiliary_client
        from agent.context_compressor import ContextCompressor

        _write_config(tmp_path, {
            "model": {"default": "gpt-5.6-luna", "provider": "custom"},
            "auxiliary": {
                "compression": {"provider": "custom", "model": "gpt-5.6-luna"},
            },
        })
        requests = []
        output_item = SimpleNamespace(
            type="message",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="compressed summary")],
        )

        def create(**kwargs):
            requests.append(kwargs)
            return iter([
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="compressed summary"),
                SimpleNamespace(type="response.output_item.done", item=output_item),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(id="response-1", status="completed", usage=None),
                ),
            ])

        real_client = SimpleNamespace(
            api_key="[REDACTED]",
            base_url="https://tianji.example.test/v1",
            responses=SimpleNamespace(create=create),
        )
        monkeypatch.setattr(auxiliary_client, "_create_openai_client", lambda **_kwargs: real_client)
        auxiliary_client._client_cache.clear()

        compressor = ContextCompressor(
            model="gpt-5.6-luna",
            provider="custom",
            base_url="https://tianji.example.test/v1",
            api_key="[REDACTED]",
            api_mode="codex_responses",
            threshold_percent=0.50,
            protect_first_n=1,
            protect_last_n=1,
            config_context_length=1_000,
            quiet_mode=True,
            tail_mode="legacy",
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third answer"},
            {"role": "user", "content": "current question"},
        ]

        assert compressor.should_compress(900)
        compressed = compressor.compress(messages, current_tokens=900)

        assert compressor.compression_count == 1
        assert len(requests) == 1
        assert requests[0]["stream"] is True
        assert requests[0]["model"] == "gpt-5.6-luna"
        assert any("compressed summary" in message.get("content", "") for message in compressed)
        assert compressor._last_summary_fallback_used is False
        telemetry = compressor._last_compression_telemetry
        assert telemetry is not None
        assert telemetry["aux_provider"] == "custom"
        assert telemetry["aux_model"] == "gpt-5.6-luna"
        assert telemetry["time_to_first_progress_ms"] is not None
        assert telemetry["failure_class"] is None
