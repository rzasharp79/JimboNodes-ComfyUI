# Replicate API Node Pattern

## Architecture
- No `replicate` Python package needed -- use `urllib.request` from stdlib
- Endpoint: POST `https://api.replicate.com/v1/models/{owner}/{model}/predictions`
- Auth: Bearer token from env var `REPLICATE_API_TOKEN`
- Headers: `Authorization`, `Content-Type: application/json`, `Prefer: wait`

## Flow
1. Validate env token exists
2. Build payload: `{"input": {...}}`
3. POST to create prediction
4. Check status: if `succeeded`, done; if `starting`/`processing`, poll `urls.get`
5. Poll with GET until `succeeded`, `failed`, or `canceled`
6. Extract output URL(s) from `output` field
7. Download image and convert to tensor

## Image Upload to Replicate
- Convert ComfyUI IMAGE tensor to PNG bytes via PIL
- Base64 encode and wrap as `data:image/png;base64,...` data URI
- Send in the `image_input` array field

## Polling Config
- Interval: 2 seconds
- Timeout: 300 seconds (5 min)
- Use `Prefer: wait` header to let server hold connection (reduces polls)

## Error Handling
- Missing token: EnvironmentError with instructions
- Empty prompt: ValueError
- HTTP errors: Read response body for details
- Timeout: RuntimeError with retry suggestion
- No output: RuntimeError with full response dump

## Output Handling
- Output can be string (single URL) or list of strings (multiple URLs)
- Download first URL, convert to PIL RGB, then to tensor
- Return shape: (1, H, W, 3) float32 [0,1]
