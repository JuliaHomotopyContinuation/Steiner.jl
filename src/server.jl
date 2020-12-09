export start_server

using Bukdu
using HTTP.Messages: setheader

struct ConicsController <: ApplicationController
    conn::Conn
end

function take_options(c::ConicsController)
    req = c.conn.request
    # @info :req_headers req.headers
    # @info :req_method_target (req.method, req.target)
    setheader(req.response, "Access-Control-Allow-Origin" => "*")
    setheader(req.response, "Access-Control-Allow-Headers" => "Content-Type")
    setheader(req.response, "Content-Length" => "0")
    setheader(req.response, "X-Content-Type-Options" => "nosniff")
    nothing
end


function solve_conics(c::ConicsController)
    req = c.conn.request
    setheader(req.response, "Access-Control-Allow-Origin" => "*")
    setheader(req.response, "Access-Control-Allow-Headers" => "Content-Type")
    if c.params.conics !== nothing

        render(JSON, handle_solve_conics(c.params.conics))
    else
        render(JSON, "{\"OK\": 1}")
    end
end

"""
    start_server(port=3264)

Start a server at the given `port`.
"""
function start_server(port=3264; async=false, host="localhost")
    routes() do
        Bukdu.options("/conics", ConicsController, take_options)
        post("/conics", ConicsController, solve_conics)
        plug(Plug.Parsers, [:json])
        plug(Plug.Static, at="/", from=normpath(@__DIR__, "..", "public"))
    end

    if async
        Bukdu.start(port)
    else
        @sync begin
            @async wait(Bukdu.start(port; host=host))
        end
    end
end
