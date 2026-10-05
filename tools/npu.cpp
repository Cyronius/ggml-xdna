// npu: runs a llama.cpp program with the NPU add-on, with nothing to set.
//
//   npu [npu's options] llama-server [its arguments]   (or: npu server ...)
//
// npu's options (--memory-gb, --min-chunk) come before the program's name,
// so they never mix with the program's own, and reach the add-on as its
// variables.
//
// It points llama.cpp at ggml-xdna.dll next to itself, asks llama.cpp
// whether the NPU is usable, and if it is adds -dev XDNA0,Vulkan0 -ts 0,1
// (and -ub 2048 -b 2048 for the programs people chat with). If it isn't, it adds
// nothing, so llama.cpp runs on the GPU as usual, and says why. With no model
// given, it lists the models it finds and asks which. What the user typed is
// passed on exactly as typed, and anything they set themselves is kept.
//
// Traces: XDNA-LAUNCHER

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// to stderr, where llama.cpp logs too
void put(const std::wstring & line) {
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode, n;
    if (GetConsoleMode(h, &mode)) {
        WriteConsoleW(h, line.c_str(), (DWORD) line.size(), &n, nullptr);
        return;
    }
    const int len = WideCharToMultiByte(CP_UTF8, 0, line.c_str(), (int) line.size(), nullptr, 0, nullptr, nullptr);
    std::string u((size_t) len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, line.c_str(), (int) line.size(), u.data(), len, nullptr, nullptr);
    WriteFile(h, u.data(), (DWORD) u.size(), &n, nullptr);
}

void say(const std::wstring & s) { put(s + L"\n"); }

std::wstring env(const wchar_t * name) {
    const DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (!n) return {};
    std::wstring v(n, L'\0');
    v.resize(GetEnvironmentVariableW(name, v.data(), n));
    return v;
}

std::wstring lower(std::wstring s) {
    for (wchar_t & c : s) c = (wchar_t) std::towlower(c);
    return s;
}

std::wstring widen(const std::string & s) {
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), len);
    return w;
}

std::string narrow(const std::wstring & w) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t) len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), len, nullptr, nullptr);
    return s;
}

// The command line after its first `skip` arguments, exactly as typed,
// split the way the C runtime splits it: argv[0] runs to its closing quote
// or the first blank; after that, quotes toggle, and backslashes escape a
// quote only in odd numbers.
const wchar_t * after_args(const wchar_t * p, int skip) {
    for (int i = 0; i < skip; i++) {
        while (*p == L' ' || *p == L'\t') p++;
        if (i == 0) {
            if (*p == L'"') {
                p++;
                while (*p && *p != L'"') p++;
                if (*p) p++;
            } else {
                while (*p && *p != L' ' && *p != L'\t') p++;
            }
            continue;
        }
        bool quoted = false;
        while (*p && (quoted || (*p != L' ' && *p != L'\t'))) {
            if (*p == L'\\') {
                const wchar_t * q = p;
                while (*q == L'\\') q++;
                if (*q == L'"') {
                    if ((q - p) % 2 == 0) quoted = !quoted;  // even: the quote is a quote; odd: a literal "
                    p = q + 1;
                } else {
                    p = q;
                }
                continue;
            }
            if (*p == L'"') {
                if (quoted && p[1] == L'"') {  // "" inside quotes: a literal quote
                    p += 2;
                    continue;
                }
                quoted = !quoted;
            }
            p++;
        }
    }
    while (*p == L' ' || *p == L'\t') p++;
    return p;
}

fs::path exe_dir() {
    wchar_t buf[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) std::size(buf));
    return fs::path(std::wstring(buf, n)).parent_path();
}

bool is_file(const fs::path & p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

//
// what the user already set
//

// llama.cpp reads "--batch_size" as "--batch-size"
std::wstring flag_name(std::wstring a) {
    if (a.rfind(L"--", 0) == 0) std::replace(a.begin(), a.end(), L'_', L'-');
    return a;
}

bool given(const std::vector<std::wstring> & args, std::initializer_list<const wchar_t *> flags,
           std::initializer_list<const wchar_t *> envs) {
    for (const std::wstring & a : args)
        for (const wchar_t * f : flags)
            if (flag_name(a) == f) return true;
    for (const wchar_t * e : envs)
        if (!env(e).empty()) return true;
    return false;
}

//
// is the NPU usable?
//

struct npu_check {
    bool usable = false;
    std::wstring why;  // when not
};

// Asks llama.cpp for its devices, with the add-on loaded, and reads the
// add-on's own line when it isn't offering the NPU.
npu_check check_npu(const fs::path & dir) {
    fs::path lister = dir / L"llama-server.exe";
    if (!is_file(lister)) lister = dir / L"llama-cli.exe";
    if (!is_file(lister)) return { false, L"no llama-server.exe next to npu.exe to ask" };

    SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return { false, L"couldn't ask llama.cpp" };
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi = {};
    std::wstring cmd = L"\"" + lister.wstring() + L"\" --list-devices";
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        return { false, L"couldn't start " + lister.filename().wstring() };
    }
    std::string out;
    char buf[4096];
    DWORD n;
    while (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n) out.append(buf, n);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, 30000);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    if (out.find("\n  XDNA0:") != std::string::npos) return { true, {} };
    const std::string tag = "xdna: not offering XDNA0: ";
    const size_t at = out.find(tag);
    if (at != std::string::npos) {
        const size_t end = out.find_first_of("\r\n", at);
        return { false, widen(out.substr(at + tag.size(), end == std::string::npos ? std::string::npos : end - at - tag.size())) };
    }
    return { false, L"the add-on didn't load (GGML_BACKEND_PATH=" + env(L"GGML_BACKEND_PATH") + L")" };
}

//
// the model menu
//

struct model {
    fs::path path;
    std::wstring name, where;
    uintmax_t size = 0;
};

fs::path state_dir() { return fs::path(env(L"LOCALAPPDATA")) / L"ggml-xdna"; }

// Every .gguf that loads as a model, in the usual places. Vision add-ons
// (mmproj-*) don't load on their own, and a model split over several files
// is listed by its first.
std::vector<model> find_models(const fs::path & dir) {
    const fs::path home = env(L"USERPROFILE");
    const std::wstring llama_cache = env(L"LLAMA_CACHE"), hf_home = env(L"HF_HOME");
    const std::vector<std::pair<fs::path, std::wstring>> roots = {
        { dir / L"models", L"models" },
        { home / L".lmstudio" / L"models", L"LM Studio" },
        { home / L".cache" / L"lm-studio" / L"models", L"LM Studio" },
        { llama_cache.empty() ? fs::path(env(L"LOCALAPPDATA")) / L"llama.cpp" : fs::path(llama_cache), L"llama.cpp" },
        { hf_home.empty() ? home / L".cache" / L"huggingface" / L"hub" : fs::path(hf_home) / L"hub", L"Hugging Face" },
    };
    static const std::wregex split(L"-(\\d{5})-of-\\d{5}$", std::regex::icase);
    std::vector<model> found;
    for (const auto & [root, where] : roots) {
        std::error_code ec;
        if (!fs::is_directory(root, ec)) continue;
        std::vector<model> here;
        for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (it.depth() > 8) {
                it.disable_recursion_pending();
                continue;
            }
            const fs::path & p = it->path();
            if (lower(p.extension().wstring()) != L".gguf" || !is_file(p)) continue;
            const std::wstring stem = p.stem().wstring();
            if (lower(stem).rfind(L"mmproj", 0) == 0) continue;
            std::wsmatch m;
            if (std::regex_search(stem, m, split) && m[1].str() != L"00001") continue;
            // who published it, when the folder says: LM Studio keeps
            // <publisher>/<model>/, the Hugging Face cache models--<publisher>--<model>/
            std::wstring label = where;
            const fs::path rel = p.lexically_relative(root);
            if (rel.begin() != rel.end() && std::next(rel.begin()) != rel.end()) {
                std::wstring top = rel.begin()->wstring();
                if (top.rfind(L"models--", 0) == 0) top = top.substr(8, top.find(L"--", 8) - 8);
                label += L": " + top;
            }
            std::error_code se;
            here.push_back({ p, stem, label, fs::file_size(p, se) });
        }
        std::sort(here.begin(), here.end(), [](const model & a, const model & b) { return lower(a.name) < lower(b.name); });
        found.insert(found.end(), here.begin(), here.end());
    }
    return found;
}

std::wstring size_text(uintmax_t b) {
    wchar_t s[32];
    if (b < 1000000000ull) swprintf(s, std::size(s), L"%llu MB", (unsigned long long) (b / 1000000));
    else swprintf(s, std::size(s), L"%.1f GB", (double) b / 1e9);
    return s;
}

std::wstring padded(std::wstring s, size_t w, bool left = true) {
    if (s.size() > w) s = s.substr(0, w - 1) + L"~";
    const std::wstring pad(w - s.size(), L' ');
    return left ? s + pad : pad + s;
}

// A router preset listing every model found, for "all of them".
fs::path write_preset(const std::vector<model> & models) {
    const fs::path p = state_dir() / L"models.ini";
    std::ofstream f(p, std::ios::binary);
    std::vector<std::wstring> used;
    for (const model & m : models) {
        std::wstring name = m.name;
        for (int i = 2; std::find(used.begin(), used.end(), lower(name)) != used.end(); i++)
            name = m.name + L"-" + std::to_wstring(i);
        used.push_back(lower(name));
        f << "[" << narrow(name) << "]\nmodel = " << narrow(m.path.wstring()) << "\n\n";
    }
    return f ? p : fs::path();
}

enum class pick_kind { model, all, quit, none };
struct pick {
    pick_kind kind = pick_kind::none;
    fs::path path;
};

pick choose_model(const fs::path & dir, bool offer_all) {
    const std::vector<model> models = find_models(dir);
    if (models.empty()) {
        say(L"npu: no models found. Looked in: a models folder next to npu.exe, LM Studio's models folder, llama.cpp's "
            L"download folder, and the Hugging Face download folder. Give one with -m <file.gguf>.");
        return { pick_kind::none, {} };
    }
    std::error_code ec;
    fs::create_directories(state_dir(), ec);
    fs::path last;
    {
        std::ifstream f(state_dir() / L"last-model.txt", std::ios::binary);
        std::string line;
        if (std::getline(f, line)) last = widen(line);
        if (!last.empty() && !is_file(last)) last.clear();
    }

    size_t w = 0;
    for (const model & m : models) w = std::max(w, m.name.size());
    w = std::min<size_t>(w, 50);
    say(L"Models found:");
    if (offer_all) say(L"   0  all of them: llama-server's router, which loads each model when it's asked for");
    for (size_t i = 0; i < models.size(); i++)
        say(padded(std::to_wstring(i + 1), 4, false) + L"  " + padded(models[i].name, w) + L"  " +
            padded(size_text(models[i].size), 8, false) + L"  " + models[i].where);

    for (;;) {
        std::wstring prompt = L"Pick a number (";
        if (!last.empty()) prompt += L"Enter: " + last.stem().wstring() + L"; ";
        prompt += L"q: quit): ";
        put(prompt);
        char line[64];
        if (!fgets(line, sizeof(line), stdin)) return { pick_kind::quit, {} };
        std::string s(line);
        s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '\r' || c == '\n' || c == ' '; }), s.end());
        if (s == "q" || s == "Q") return { pick_kind::quit, {} };
        fs::path chosen;
        if (s.empty() && !last.empty()) {
            chosen = last;
        } else if (!s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }) && s.size() < 6) {
            const size_t k = (size_t) std::stoul(s);
            if (k == 0 && offer_all) {
                const fs::path ini = write_preset(models);
                if (!ini.empty()) return { pick_kind::all, ini };
                say(L"npu: couldn't write " + (state_dir() / L"models.ini").wstring());
                continue;
            }
            if (k >= 1 && k <= models.size()) chosen = models[k - 1].path;
        }
        if (chosen.empty()) continue;
        std::ofstream f(state_dir() / L"last-model.txt", std::ios::binary);
        f << narrow(chosen.wstring()) << "\n";
        return { pick_kind::model, chosen };
    }
}

//
// running the program
//

BOOL WINAPI ignore_ctrl_c(DWORD type) {
    // the program gets Ctrl+C too, and stops itself; we wait for it
    return type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT;
}

int run(const fs::path & exe, const std::wstring & args) {
    std::wstring cmd = L"\"" + exe.wstring() + L"\"" + args;
    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
        say(L"npu: couldn't start " + exe.wstring());
        return 1;
    }
    // if npu is closed or killed, the program (and anything it started) goes too
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
        AssignProcessToJobObject(job, pi.hProcess);
    }
    SetConsoleCtrlHandler(ignore_ctrl_c, TRUE);
    ResumeThread(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (job) CloseHandle(job);
    return (int) code;
}

void usage() {
    say(L"npu runs a llama.cpp program with prompts read on the NPU.\n"
        L"\n"
        L"  npu [npu's options] llama-server [its arguments]      (or: npu server ...)\n"
        L"  npu llama-cli -m model.gguf\n"
        L"\n"
        L"It points llama.cpp at the NPU add-on next to it and, when the NPU can run, adds\n"
        L"-dev XDNA0,Vulkan0 -ts 0,1 (plus -ub 2048 -b 2048 for llama-server, llama-cli\n"
        L"and llama-completion). With no model given, it lists the models it finds.\n"
        L"Anything you set yourself is kept.\n"
        L"\n"
        L"npu's options, before the program's name:\n"
        L"  --memory-gb N   the most memory the NPU's copies of the weights may take, in GB\n"
        L"                  (0.5 and the like work; 0 keeps the NPU out). Default: the\n"
        L"                  memory free once the model is loaded, less 4 GB or a tenth of\n"
        L"                  the machine's memory, whichever is larger, and at most 20 GB:\n"
        L"                  past about 26 GB the NPU's copies can go bad. Layers that\n"
        L"                  don't fit stay on the GPU.\n"
        L"  --min-chunk N   the smallest piece of a prompt, in tokens, the NPU takes.\n"
        L"                  Default 1024; smaller pieces go to the GPU, which read them\n"
        L"                  faster in our tests. 512 also uses the NPU with llama.cpp's\n"
        L"                  default chunk size.");
}

// npu's own options, before the program's name.
struct options {
    std::wstring memory_gb;  // as typed; empty when not given
    std::wstring min_chunk;
};

// A size in GB: a number, 0 or more.
bool is_gb(const std::wstring & s) {
    if (s.empty()) return false;
    wchar_t * end = nullptr;
    const double v = wcstod(s.c_str(), &end);
    return *end == 0 && v >= 0 && v < 1e6;
}

// A token count: a whole number, 1 or more.
bool is_tokens(const std::wstring & s) {
    if (s.empty() || s.size() > 9) return false;
    return std::all_of(s.begin(), s.end(), [](wchar_t c) { return c >= L'0' && c <= L'9'; }) && std::stol(s) >= 1;
}

// Reads npu's options from argv[1] on, as "--name value" or "--name=value".
// Returns where the program's name is, 0 when help was asked for, or -1
// after saying what's wrong.
int read_options(int argc, LPWSTR * argv, options & o) {
    int i = 1;
    for (; i < argc; i++) {
        const std::wstring a = argv[i];
        if (lower(a) == L"-h" || lower(a) == L"--help") return 0;
        if (a.rfind(L"-", 0) != 0) break;
        const size_t eq = a.find(L'=');
        const std::wstring name = lower(a.substr(0, eq));
        if (name != L"--memory-gb" && name != L"--min-chunk") {
            say(L"npu: unknown option " + a + L" (npu's options go before the program's name; npu -h lists them)");
            return -1;
        }
        std::wstring value;
        if (eq != std::wstring::npos) value = a.substr(eq + 1);
        else if (i + 1 < argc) value = argv[++i];
        const std::wstring not_this = value.empty() ? std::wstring() : L", not \"" + value + L"\"";
        if (name == L"--memory-gb") {
            if (!is_gb(value)) {
                say(L"npu: --memory-gb takes a size in GB, such as 20 or 0.5 (0 keeps the NPU out)" + not_this);
                return -1;
            }
            o.memory_gb = value;
        } else {
            if (!is_tokens(value)) {
                say(L"npu: --min-chunk takes a number of tokens, such as 512 or 2048" + not_this);
                return -1;
            }
            o.min_chunk = value;
        }
    }
    return i;
}

} // namespace

int wmain() {
    int argc = 0;
    LPWSTR * argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return 1;
    options opt;
    const int at = read_options(argc, argv, opt);  // argv[at]: the program
    if (at < 0) return 1;
    if (at == 0 || at >= argc) {
        usage();
        return at == 0 ? 0 : 1;
    }
    const fs::path dir = exe_dir();

    // the program: "llama-server", "llama-server.exe" or "server"
    std::wstring tool = argv[at];
    if (lower(tool).size() > 4 && lower(tool).substr(lower(tool).size() - 4) == L".exe") tool.resize(tool.size() - 4);
    fs::path exe = dir / (tool + L".exe");
    if (!is_file(exe) && is_file(dir / (L"llama-" + tool + L".exe"))) {
        tool = L"llama-" + tool;
        exe = dir / (tool + L".exe");
    }
    if (!is_file(exe)) {
        say(L"npu: there's no " + tool + L".exe next to npu.exe (" + dir.wstring() + L")");
        return 1;
    }
    tool = lower(tool);
    const std::vector<std::wstring> args(argv + at + 1, argv + argc);
    std::wstring added;

    // the add-on, unless the user points llama.cpp elsewhere
    if (env(L"GGML_BACKEND_PATH").empty()) {
        const fs::path dll = dir / L"ggml-xdna.dll";
        if (is_file(dll)) SetEnvironmentVariableW(L"GGML_BACKEND_PATH", dll.c_str());
    }
    // npu's options reach the add-on as its variables, which the program
    // and anything it starts (the router's per-model servers) inherit
    std::wstring set;
    if (!opt.memory_gb.empty()) {
        SetEnvironmentVariableW(L"GGML_XDNA_MAX_COPY_GB", opt.memory_gb.c_str());
        set += L"; NPU weight copies limited to " + opt.memory_gb + L" GB";
    }
    if (!opt.min_chunk.empty()) {
        SetEnvironmentVariableW(L"GGML_XDNA_MIN_BATCH", opt.min_chunk.c_str());
        set += L"; NPU takes chunks of " + opt.min_chunk + L" tokens or more";
    }

    // which programs take which of our additions
    static const std::vector<std::wstring> common = { L"llama-server", L"llama-cli", L"llama-completion",
                                                      L"llama-perplexity", L"llama-batched-bench", L"llama-mtmd-cli",
                                                      L"llama-imatrix", L"llama-embedding" };
    const bool bench = tool == L"llama-bench";
    const bool takes_dev = bench || std::find(common.begin(), common.end(), tool) != common.end();
    const bool chat = tool == L"llama-server" || tool == L"llama-cli" || tool == L"llama-completion";
    const bool info = given(args, { L"-h", L"--help", L"--usage", L"--version", L"--list-devices", L"--completion-bash",
                                    L"--license" }, {});

    if (takes_dev && !info && !given(args, { L"-dev", L"--device" }, { L"LLAMA_ARG_DEVICE" })) {
        const npu_check c = check_npu(dir);
        if (c.usable) {
            added += bench ? L" -dev XDNA0/Vulkan0" : L" -dev XDNA0,Vulkan0";
            // No layers on XDNA0, as llama.cpp would pick anyway (it reports no
            // free memory). Said outright, it also stops llama.cpp's memory
            // fitting before its layer search, which divides by zero on
            // mixture-of-experts models with XDNA0 listed: XDNA0 shares the
            // GPU's memory, so its use never changes as layers move onto it.
            // llama-bench doesn't fit unless asked.
            if (!bench && !given(args, { L"-ts", L"--tensor-split" }, { L"LLAMA_ARG_TENSOR_SPLIT" }))
                added += L" -ts 0,1";
            if (chat && !given(args, { L"-b", L"--batch-size", L"-ub", L"--ubatch-size" },
                               { L"LLAMA_ARG_BATCH", L"LLAMA_ARG_UBATCH" }))
                added += L" -ub 2048 -b 2048";
            say(L"npu: prompts on the NPU (added" + added + set + L")");
        } else {
            say(L"npu: running on the GPU only: " + c.why);
        }
    }

    // a model, if none was given and someone is there to pick one
    DWORD mode;
    const bool loads_model = takes_dev && tool != L"llama-mtmd-cli";
    if (loads_model && !info && GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode) &&
        !given(args, { L"-m", L"--model", L"-mu", L"--model-url", L"-hf", L"-hfr", L"--hf-repo", L"-dr",
                       L"--docker-repo", L"--models-dir", L"--models-preset" },
               { L"LLAMA_ARG_MODEL", L"LLAMA_ARG_MODEL_URL", L"LLAMA_ARG_HF_REPO", L"LLAMA_ARG_DOCKER_REPO",
                 L"LLAMA_ARG_MODELS_DIR", L"LLAMA_ARG_MODELS_PRESET" })) {
        const pick p = choose_model(dir, tool == L"llama-server");
        if (p.kind == pick_kind::quit) return 0;
        if (p.kind == pick_kind::none) return 1;
        added += (p.kind == pick_kind::all ? L" --models-preset \"" : L" -m \"") + p.path.wstring() + L"\"";
    }

    const std::wstring rest = after_args(GetCommandLineW(), at + 1);
    LocalFree(argv);
    return run(exe, added + (rest[0] ? L" " + rest : std::wstring()));
}
