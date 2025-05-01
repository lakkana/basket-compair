using System;
using System.IO;
using System.Text.Json;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;
using System.Net.Http;
using System.Net.Http.Headers;

public static class VehicleRecommendationFunction
{
    private static readonly HttpClient httpClient = new HttpClient();
    private static JArray vehicleData;

    static VehicleRecommendationFunction()
    {
        string filePath = Path.Combine(Environment.CurrentDirectory, "hackathon10_input_cleaned_updated.json");
        string json = File.ReadAllText(filePath);
        vehicleData = JArray.Parse(json);
    }

    [FunctionName("RecommendVehicles")]
    public static async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = "recommend")] HttpRequest req,
        ILogger log)
    {
        try
        {
            var body = await new StreamReader(req.Body).ReadToEndAsync();
            var data = JsonSerializer.Deserialize<Dictionary<string, string>>(body);

            string make = data["make"];
            string model = data["model"];
            string version = data["version_name"];

            var benchmark = vehicleData
                .FirstOrDefault(v =>
                    string.Equals(v["make"]?.ToString(), make, StringComparison.OrdinalIgnoreCase) 
                    && string.Equals(v["model"]?.ToString(), model, StringComparison.OrdinalIgnoreCase)
                    && Similarity(v["version_name"]?.ToString() ?? "", version) >= 0.6
                );

            if (benchmark == null)
                return new BadRequestObjectResult("Benchmark vehicle not found.");

            var similarVehicles = FilterSimilarVehicles(benchmark, strict: true);
            if (similarVehicles.Count < 10)
                similarVehicles = FilterSimilarVehicles(benchmark, strict: false);

            string gptPrompt = GeneratePrompt(benchmark, similarVehicles);

            string gptResponse = await CallOpenAI(gptPrompt);

            string uidList = await ExtractUIDs(gptResponse);
            var uidArray = uidList.Split(",").Select(uid => uid.Trim()).ToHashSet();

            var bestMatches = similarVehicles
                .Where(v => uidArray.Contains(v["uid"]?.ToString()))
                .GroupBy(v => v["model"]?.ToString())
                .Select(g => g.First())
                .ToList();

            foreach (var item in bestMatches)
            {
                item["GPT_Reasoning"] = gptResponse;
            }

            return new OkObjectResult(bestMatches);
        }
        catch (Exception ex)
        {
            return new ObjectResult(new { error = ex.Message }) { StatusCode = 500 };
        }
    }
    private static double Similarity(string s1, string s2)
    {
        if (string.IsNullOrWhiteSpace(s1) || string.IsNullOrWhiteSpace(s2)) return 0;
        int distance = LevenshteinDistance(s1.ToLower(), s2.ToLower());
        return 1.0 - (double)distance / Math.Max(s1.Length, s2.Length);
    }

    private static int LevenshteinDistance(string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[,] d = new int[n + 1, m + 1];

        if (n == 0) return m;
        if (m == 0) return n;

        for (int i = 0; i <= n; d[i, 0] = i++) { }
        for (int j = 0; j <= m; d[0, j] = j++) { }

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = t[j - 1] == s[i - 1] ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost
                );
            }
        }

        return d[n, m];
    }


    private static List<JToken> FilterSimilarVehicles(JToken benchmark, bool strict)
    {
        int lenDelta = strict ? 200 : 300;
        int widthDelta = strict ? 80 : 120;
        int heightDelta = strict ? 200 : 300;
        double priceRange = strict ? 0.1 : 0.2;
        double powerRange = strict ? 0.1 : 0.2;

        return vehicleData
            .Where(v =>
                v["make"]?.ToString() != benchmark["make"]?.ToString() &&
                v["bodystyle"]?.ToString() == benchmark["bodystyle"]?.ToString() &&
                v["powertrain_type"]?.ToString() == benchmark["powertrain_type"]?.ToString() &&
                Math.Abs((double)v["overall_length"] - (double)benchmark["overall_length"]) <= lenDelta &&
                Math.Abs((double)v["overall_width"] - (double)benchmark["overall_width"]) <= widthDelta &&
                Math.Abs((double)v["overall_height"] - (double)benchmark["overall_height"]) <= heightDelta &&
                Math.Abs((double)v["retail_price"] - (double)benchmark["retail_price"]) <= (double)benchmark["retail_price"] * priceRange &&
                Math.Abs((double)v["max_power_hp_ps"] - (double)benchmark["max_power_hp_ps"]) <= (double)benchmark["max_power_hp_ps"] * powerRange &&
                v["transmission_type"]?.ToString() == benchmark["transmission_type"]?.ToString() &&
                v["fuel_type"]?.ToString() == benchmark["fuel_type"]?.ToString()
            )
            .ToList();
    }

    private static string GeneratePrompt(JToken benchmark, List<JToken> similar)
    {
        var lines = new List<string>
        {
            $"Make: {benchmark["make"]}",
            $"Model: {benchmark["model"]}",
            $"Version: {benchmark["version_name"]}",
            $"Body Style: {benchmark["bodystyle"]}",
            $"Powertrain Type: {benchmark["powertrain_type"]}",
            $"Transmission Type: {benchmark["transmission_type"]}",
            $"Fuel Type: {benchmark["fuel_type"]}",
            $"Length: {benchmark["overall_length"]} mm",
            $"Width: {benchmark["overall_width"]} mm",
            $"Height: {benchmark["overall_height"]} mm",
            $"Retail Price: {benchmark["retail_price"]} USD",
            $"Max Power: {benchmark["max_power_hp_ps"]} HP/PS"
        };

        var table = string.Join("\n", similar.Select(v =>
            $"{v["version_name"],-25} {v["make"],-15} {v["model"],-15} {v["uid"],-10} {v["transmission_type"],-10} {v["fuel_type"],-10} {v["retail_price"],-10} {v["max_power_hp_ps"]}"
        ));

        return $@"
You are an AI automotive data expert. Here is the benchmark and candidate list.

### Benchmark:
{string.Join("\n", lines)}

### Similar Vehicles:
{table}

### Task:
1. Rank top 10 vehicles.
2. Only 1 version per model.
3. Output Make, Model, Version, UID, Transmission, Fuel, Price, Power, Reasoning.
";
    }

    private static async Task<string> CallOpenAI(string prompt)
    {
        string endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        string deployment = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT");
        string apiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

        var request = new
        {
            messages = new[]
            {
                new { role = "system", content = "You are an automotive data assistant." },
                new { role = "user", content = prompt }
            },
            temperature = 0.7,
            max_tokens = 2500
        };

        var requestMessage = new HttpRequestMessage(HttpMethod.Post,
            $"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-06-01");

        requestMessage.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        requestMessage.Content = new StringContent(JsonSerializer.Serialize(request));
        requestMessage.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");

        var response = await httpClient.SendAsync(requestMessage);
        response.EnsureSuccessStatusCode();
        string result = await response.Content.ReadAsStringAsync();

        var json = JsonDocument.Parse(result);
        return json.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").ToString();
    }

    private static async Task<string> ExtractUIDs(string gptResponse)
    {
        string prompt = $@"
Here is a response about vehicles:
{gptResponse}

Extract ONLY the UIDs, comma-separated.
";

        return await CallOpenAI(prompt);
    }
}
